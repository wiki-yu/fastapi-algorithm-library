from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from imutils.video import FPS

from app.libs.utils.general import increment_path


class Vgg_RollAverage_RealTime(object):
    def __init__(self, size: int = 224, roll_len: int = 128):
        self.size = size
        self.roll_len = roll_len

    def preprocess_image(self, img):
        """
        Normalize and resize the image to feed into the model

        Args:
            img : extracted image from the video with a size of (634 x 640)

        Returns:
            Normalized data (Z-score) between [0,1] in shape of model input
        """
        img = cv2.resize(img, (self.size, self.size))
        img = img/255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return img

    def real_time_prediction(
        self,
        video_path: Path,
        weights_filepath: Path,
        video_name: str = None,
        roll_average: bool = True
    ) -> Path:
        """
        Make real time prediction on input video

        Args:
            video_path (Path) : path to input video,
            weights_filepath (Path) : path to trained weights,
            video_name (str) : optional name of input video,
            roll_average (bool) : decides if prediction is made based on past
            few (roll_len) frames or based on single frame

        Returns:
            The path to the results .csv file.
            :Prints elasped time, time taken to process the video,
            and approximate FPS, computed frames per second
        """
        if video_name:
            video_name = video_name.rsplit('/', maxsplit=1)[-1].split('.')[0]
        else:
            video_name = video_path.as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]

        class_names = ['away', 'clean_tray', 'count_pills', 'drawer', 'scan_papers_computer']
        cap = cv2.VideoCapture(video_path.as_posix())
        model = load_model(weights_filepath)
        out_df = pd.DataFrame(columns=['frame_index', 'pred_cls', 'pred_prob'])

        frame_index = 0
        Q = deque(maxlen=self.roll_len)
        fps = FPS().start()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = self.preprocess_image(frame)
            pred_probs = model.predict(np.expand_dims(img, axis=0))

            if roll_average:    # Prediction is made based on past few (roll_len) frames
                Q.append(pred_probs)
                rolling_probs = np.array(Q).mean(axis=0)
                rolling_prob = max(rolling_probs[0])
                index = np.argmax(rolling_probs, axis=1)[0]
                pred_cls = class_names[index]
                out_df = out_df.append(
                    {
                        'frame_index': frame_index,
                        'pred_cls': pred_cls,
                        'pred_prob': rolling_prob
                    },
                    ignore_index=True
                )
                frame_index += 1
                fps.update()
            else:   # Prediction is made based on single frame
                pred_prob = max(pred_probs[0])
                index = np.argmax(pred_probs, axis=1)[0]
                pred_cls = class_names[index]
                out_df = out_df.append(
                    {
                        'frame_index': frame_index,
                        'pred_cls': pred_cls,
                        'pred_prob': pred_prob
                    },
                    ignore_index=True
                )
                frame_index += 1
                fps.update()
        fps.stop()

        print(f"[INFO] elasped time: {fps.elapsed():.2f}")
        print(f"[INFO] approx. FPS: {fps.fps():.2f}")

        cap.release()
        cv2.destroyAllWindows()

        # Save results
        save_dir = increment_path(
            Path(__file__).joinpath(
                f'../../../../results/rolling_average/{video_name}').resolve(),
            sep='_',
            mkdir=True
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir.resolve().joinpath(f'{video_name}.csv')

        out_df.to_csv(save_path)

        return save_path


if __name__ == '__main__':
    # Path where the video is saved (modify the video path if requires)
    video_path_test = 'v-scan_papers_computer-clip2-1.mp4'
    # Path where the model is saved (modify the weights path if requires)
    weights_filepath_test = './result_dir/C3D_tf_vgg_test_ines_3.hdf5'

    data_generator_object = Vgg_RollAverage_RealTime(size=224, roll_len=128)
    data_generator_object.real_time_prediction(
        video_path=video_path_test,
        weights_filepath=weights_filepath_test,
        roll_average=True
    )
