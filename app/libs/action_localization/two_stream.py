import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from imutils.video import FPS

from app.libs.utils.general import increment_path


class Two_Stream_RealTime(object):
    def __init__(self, size: int = 224, roll_len: int = 16):
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

    def write_image(
        self, image, pred_probs_rgb, pred_probs_of,
        output, pred_cls, combined_prob, fps, display=False
    ):
        """
        Write prediction results (probability & class) onto images. Writes video file output inplace.

        Args:
            image : extracted image from the video where predictions will be written
            pred_probs_rgb : array (1 x no. action class) corresponds to the prediction from rgb image
            pred_probs_of : array (1 x no. action class) corresponds to the prediction from of image
            output : mp4 file where output will be written
            pred_cls : predicted class based on combined probability (rgb+of)
            combined_prob : probability (rgb+of) obtained combing two different streams
            fps : calculated fps
            display : decides if we want to visualize/display each image after predictions are made
        """
        pred_prob_rgb = max(pred_probs_rgb[0])
        pred_prob_of = max(pred_probs_of[0])

        image = cv2.putText(
            image, 'Ground_truth:scan_papers_computer', (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        image = cv2.putText(image, 'Prediction:' + pred_cls, (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        image = cv2.putText(
            image, 'RGB_prob:'f'{pred_prob_rgb:0.6f}', (5, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        image = cv2.putText(
            image, 'OF_prob:'f'{pred_prob_of:0.6f}', (5, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        image = cv2.putText(
            image, 'Combined_prob:'f'{combined_prob:0.6f}', (5, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        image = cv2.putText(image, 'FPS:'f'{fps:.2f}', (5, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        output.write(image)

    def prediction_single_frame(self, class_names, frame_index, pred_probs, df):
        """
        Make prediction results based on single frame

        Args:
            class_names : list of action classes of the dataset
            frame_index : index of frames of videos
            pred_probs : array (1 x no. action class) corresponds to combined predicted probabilites (rgb+of)
            df : dataframe where results will be saved as a csv file

        Returns:
            pred_prob : maximum probability obtained from pred_probs upon single frame
            pred_cls : predicted class using combined probability (rgb+of) upon single frame
            df : output data frame
        """
        pred_prob = max(pred_probs[0])
        index = np.argmax(pred_probs, axis=1)[0]
        pred_cls = class_names[index]
        df = df.append(
            {
                'frame_index': frame_index,
                'pred_cls': pred_cls,
                'pred_prob': pred_prob
            },
            ignore_index=True
        )
        return pred_prob, pred_cls, df

    def prediction_multiple_frames(self, Q, class_names, frame_index, pred_probs, df):
        """
        Make prediction results based on past n frames

        Args:
            Q : deque that keeps storing the predictions
            class_names : list of action classes of the dataset
            frame_index : index of frames of videos
            pred_probs : array (1 x no. action class) corresponds to combined predicted probabilites (rgb+of)
            df : dataframe where results will be saved as a csv file

        Returns:
            rolling_prob : maximum probability obtained from pred_probs upon multiple frame
            pred_cls : pred_cls: predicted class using combined probability (rgb+of) upon multiple frame
            df : output data frame
        """
        Q.append(pred_probs)
        rolling_probs = np.array(Q).mean(axis=0)
        rolling_prob = max(rolling_probs[0])
        index = np.argmax(rolling_probs, axis=1)[0]
        pred_cls = class_names[index]
        df = df.append(
            {
                'frame_index': frame_index,
                'pred_cls': pred_cls,
                'pred_prob': rolling_prob
            },
            ignore_index=True
        )
        return rolling_prob, pred_cls, df

    def real_time_prediction(
        self,
        video_path: Path,
        weight_path_rgb: Path,
        weight_path_of: Path,
        video_name: str = None,
        roll_average: bool = False
    ):
        """
        Extract RGB & OF images from the input video and make a combined real-time prediction

        Args:
            video_path (Path) : path to input video
            weight_path_rgb (Path) : path to weight file (hdf5) for model trained with RGB data
            weight_path_of (Path) : path to weight file (hdf5) for model trained with OF data
            one_frame: decides if prediction is made based on single frame
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

        # Save results
        save_dir = increment_path(
            Path(__file__).joinpath(
                f'../../../../results/two_stream/{video_name}').resolve(),
            sep='_',
            mkdir=True
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path_csv = save_dir.resolve().joinpath(f'{video_name}.csv')

        class_names = ['away', 'clean_tray', 'count_pills', 'drawer', 'scan_papers_computer']
        cap = cv2.VideoCapture(video_path.as_posix())
        output = cv2.VideoWriter(
            save_dir.resolve().joinpath(f'{video_name}.mp4').as_posix(),
            cv2.VideoWriter_fourcc(*'MP4V'),
            15, (640, 634)
        )
        model_rgb = load_model(weight_path_rgb)
        model_of = load_model(weight_path_of)
        method = cv2.optflow.calcOpticalFlowSparseToDense
        out_df = pd.DataFrame(columns=['frame_index', 'pred_cls', 'pred_prob'])

        frame_index = 0
        Q = deque(maxlen=self.roll_len)
        fps = FPS().start()
        ret, old_frame = cap.read()
        # Create HSV & make Value a constant
        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255

        # Used to record the time when we processed last frame
        prev_frame_time = 0
        # Used to record the time at which we processed current frame
        new_frame_time = 0

        while True:
            if frame_index % 20 == 0:
                print(frame_index)
            ret, frame = cap.read()
            if not ret:
                break

            image = frame    # Copy the image to write on it after prediction is made
            img = self.preprocess_image(frame)  # Use counter to deal with the 1st image
            pred_probs_rgb = model_rgb.predict(np.expand_dims(img, axis=0))

            # Calculate Optical Flow
            flow = method(old_frame, frame, None)
            # Encoding: convert the algorithm's output into Polar coordinates
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Use Hue and Value to encode the Optical Flow
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # Convert HSV image into BGR for demo
            of_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # cv2.imwrite('./OF_test_optimo/{:04}.jpg'.format(frame_index), of_img)
            of_img = self.preprocess_image(of_img)  # Use counter to deal with the 1st image
            pred_probs_of = model_of.predict(np.expand_dims(of_img, axis=0))
            pred_probs = (pred_probs_rgb + pred_probs_of)/2
            # Update the previous frame
            old_frame = frame

            if roll_average:    # Prediction is made based on past few (roll_len) frames
                pred_prob, pred_cls, out_df = self.prediction_multiple_frames(
                    Q, class_names, frame_index, pred_probs, out_df
                )

                # Calculate frame per second
                new_frame_time = time.time()
                freq = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                # Call the funcion to write and save the output video
                self.write_image(image, pred_probs_rgb, pred_probs_of, output, pred_cls, pred_prob, freq)

                frame_index += 1
                fps.update()
            else:   # Prediction is made based on single frame
                pred_prob, pred_cls, out_df = self.prediction_single_frame(
                    class_names, frame_index, pred_probs, out_df
                )

                # Calculate frame per second
                new_frame_time = time.time()
                freq = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                # Call the funcion to write and save the output video
                self.write_image(image, pred_probs_rgb, pred_probs_of, output, pred_cls, pred_prob, freq)

                frame_index += 1
                fps.update()
        fps.stop()

        print(f"[INFO] elasped time: {fps.elapsed():.2f} sec")
        print(f"[INFO] approx. FPS: {fps.fps():.2f}")

        cap.release()
        cv2.destroyAllWindows()

        out_df.to_csv(save_path_csv)

        return save_path_csv
