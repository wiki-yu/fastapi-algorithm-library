import time
import os
import cv2
# import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

from app.libs.utils.general import increment_path
from app.libs.action_localization.spatio_temporal_pytorch.core.utils import read_data_cfg, get_region_boxes, nms
from app.libs.action_localization.spatio_temporal_pytorch.cfg.cfg import parse_cfg
from app.libs.action_localization.spatio_temporal_pytorch.core.region_loss import RegionLoss
from app.libs.action_localization.spatio_temporal_pytorch.core.model import YOWO
from app.libs.action_localization.spatio_temporal_pytorch.datasets import cv2_transform


class Spatio_Temporal_Model:
    def __init__(
        self,
        video_path: Path,
        model_path_spatio_temporal: Path,
        model_path_backbone2d: Path,
        model_path_backbone3d: Path,
        video_name: str,
    ) -> None:
        self.video_path = video_path
        self.model_path_spatio_temporal = model_path_spatio_temporal
        self.model_path_backbone2d = model_path_backbone2d
        self.model_path_backbone3d = model_path_backbone3d
        self.video_name = video_name.rsplit('.', maxsplit=1)[0]

        current_path = self.get_current_dir()

        self.dataset = 'ucsp'
        self.data_cfg = current_path + '/cfg/ucsp.data'
        self.cfg_file = current_path + '/cfg/ucsp.cfg'
        self.n_classes = 4
        self.backbone_2d = 'darknet'
        self.backbone_2d_weights = self.model_path_backbone2d
        self.backbone_3d = 'resnext101'
        self.backbone_3d_weights = self.model_path_backbone3d

    def get_current_dir(self):
        return os.path.dirname(__file__)

    def action_detection(self, verbose: bool = False):
        # Dataset to use
        dataset_use = self.dataset
        assert dataset_use == 'ucsp' or dataset_use == 'ucf101-24', 'invalid dataset'

        # Configurations
        datacfg = self.data_cfg  # path for dataset of training and validation
        cfgfile = self.cfg_file  # path for cfg file
        data_options = read_data_cfg(datacfg)
        net_options = parse_cfg(cfgfile)[0]

        # GPU parameters
        gpus = data_options['gpus']  # e.g. 0,1,2,3

        # Test parameters
        batch_size = int(net_options['batch'])
        clip_duration = int(net_options['clip_duration'])
        nms_thresh = 0.4

        use_cuda = True
        seed = int(time.time())
        torch.manual_seed(seed)   # set random seed
        if use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            torch.cuda.manual_seed(seed)

        # Loss parameters
        loss_options = parse_cfg(cfgfile)[1]
        region_loss = RegionLoss()
        anchors = loss_options['anchors'].split(',')
        region_loss.anchors = [float(i) for i in anchors]
        region_loss.num_classes = int(loss_options['classes'])
        region_loss.num_anchors = int(loss_options['num'])
        region_loss.anchor_step = len(region_loss.anchors) // region_loss.num_anchors
        region_loss.object_scale = float(loss_options['object_scale'])
        region_loss.noobject_scale = float(loss_options['noobject_scale'])
        region_loss.class_scale = float(loss_options['class_scale'])
        region_loss.coord_scale = float(loss_options['coord_scale'])
        region_loss.batch = batch_size

        # Create model
        model = YOWO(self)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)  # in multi-gpu case
        model.seen = 0

        # Load spatio-temporal action localization model
        if self.model_path_spatio_temporal:
            checkpoint = torch.load(self.model_path_spatio_temporal)
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded model fscore: ", checkpoint['fscore'])

        region_loss.seen = model.seen

        num_classes = region_loss.num_classes
        anchors = region_loss.anchors
        num_anchors = region_loss.num_anchors
        conf_thresh_valid = 0.2  # 0.005
        model.eval()

        # Data preparation and inference
        video_path = self.video_path.as_posix()
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)

        # Create the folder to save the processed video
        save_dir = increment_path(
            Path(__file__).joinpath(
                f'../../../../../results/spatio_temporal/{self.video_name}').resolve(),
            sep='_',
            mkdir=True
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        # Set up video writting
        # result_video_name = f"{self.video_name}_detection.avi"
        result_video_name = f"{self.video_name}_detection.mp4"
        result_save_path = Path(
            save_dir.as_posix().rsplit('.', maxsplit=1)[0]).joinpath(result_video_name).as_posix()
        # result = cv2.VideoWriter(result_save_path, cv2.VideoWriter_fourcc(*'MJPG'), 20, (320, 240))  # size)
        result = cv2.VideoWriter(result_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (320, 240))  # size)

        count = 0
        queue = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            count += 1
            if verbose:
                if (count == 1) or (count % 100 == 0):
                    print('Count: ', count)

            if not ret:
                break
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

            if len(queue) <= 0:
                for i in range(clip_duration):
                    queue.append(frame)
            else:
                queue.append(frame)
                queue.pop(0)

            # Resize images
            imgs = [cv2_transform.resize(224, img) for img in queue]

            imgs = [cv2_transform.HWC2CHW(img) for img in imgs]  # convert image to CHW keeping BGR order.
            imgs = [img / 255.0 for img in imgs]  # image [0, 255] -> [0, 1].
            imgs = [
                np.ascontiguousarray(
                    img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
                ).astype(np.float32)
                for img in imgs
            ]

            # Concat list of images to single ndarray.
            imgs = np.concatenate([np.expand_dims(img, axis=1) for img in imgs], axis=1)
            imgs = np.ascontiguousarray(imgs)
            imgs = torch.from_numpy(imgs)
            imgs = torch.unsqueeze(imgs, 0)

            # Model inference
            with torch.no_grad():
                output = model(imgs).data
                preds = []
                # print('### model output shape: ', output.shape)  # [1, 425, 7, 7]
                all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)

                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    if verbose:
                        if (count == 1) or (count % 100 == 0):
                            print(f'boxes.shape: {np.shape(boxes)} ...')
                    for box in boxes:
                        x1 = round(float(box[0] - box[2] / 2.0) * 320.0)
                        y1 = round(float(box[1] - box[3] / 2.0) * 240.0)
                        x2 = round(float(box[0] + box[2] / 2.0) * 320.0)
                        y2 = round(float(box[1] + box[3] / 2.0) * 240.0)

                        det_conf = float(box[4])
                        for j in range((len(box) - 5) // 2):
                            cls_conf = float(box[5 + 2 * j].item())

                            if type(box[6 + 2 * j]) == torch.Tensor:
                                cls_id = int(box[6 + 2 * j].item())
                            else:
                                cls_id = int(box[6 + 2 * j])
                            prob = det_conf * cls_conf
                            preds.append([[x1, y1, x2, y2], prob, cls_id])

            for dets in preds:
                x1 = int(dets[0][0])
                y1 = int(dets[0][1])
                x2 = int(dets[0][2])
                y2 = int(dets[0][3])
                cls_score = np.array(dets[1])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                blk = np.zeros(frame.shape, np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text = []
                text_size = []
                text.append("[{:.2f}] ".format(cls_score) + 'class:' + str(dets[2]))
                text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.25, thickness=1)[0])
                coord.append((x1+3, y1+7+10))
                cv2.rectangle(
                    blk,
                    (coord[-1][0]-1, coord[-1][1]-6),
                    (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4),
                    (0, 255, 0),
                    cv2.FILLED
                )
                frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
                for t in range(len(text)):
                    cv2.putText(frame, text[t], coord[t], font, 0.25, (0, 0, 0), 1)

            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        result.release()
        cap.release()
        return result_save_path
