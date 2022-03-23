from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from imutils.video import FPS
from tensorflow.keras.preprocessing.image import img_to_array
from joblib import load
from scipy.stats import mode

from app.libs.utils.general import increment_path


class DenseNet_XGBoost(object):
    def __init__(
        self,
        input_video_path: Path,
        model_path_densenet: Path,
        model_path_xgboost: Path,
        model_path_classes: Path,
        video_name: str = None,
    ):
        self.input_video_path = input_video_path
        self.model_path_densenet = model_path_densenet
        self.model_path_xgboost = model_path_xgboost
        self.model_path_classes = model_path_classes
        self.video_name = video_name
        self.batch_size = 64
        self.time_window = 7

    def process_to_annotation_data(self, df, class_names, video_fps, min_len):
        """
        This function cleans the output data, so that there are
        no jumping frames.
        """
        j = 1  # Helper

        # Minimum qty of frames of the same task in order to
        # consider it a whole task
        min_frames = int(float(min_len) * float(video_fps) * float(0.6))

        # Initialize variables
        df["subgroup"] = (df.iloc[:, -1] != df.iloc[:, -1].shift(1)).cumsum()
        added = (
            df["subgroup"]
            .value_counts()[df["subgroup"].value_counts() < (j + 1)]
            .index.tolist()
        )

        # Modify jumping frames by considering the sourrounding frames
        # check for frames that jump (the total group of those frames are of a max of 7)
        for jj in range(min_frames):

            j = jj + 1

            df["subgroup"] = (df.iloc[:, -2] != df.iloc[:, -2].shift(1)).cumsum()
            added = (
                df["subgroup"]
                .value_counts()[df["subgroup"].value_counts() < (j + 1)]
                .index.tolist()
            )

            cnt = 0
            i_prev = 0
            i_prev_cnt = 0
            while len(added) > 0:
                added.sort()
                i = added[0]

                k = 1  # Helper
                prev = []
                after = []
                prev_yes = 0
                after_yes = 0
                if (i - k) > 0:
                    prev = [df[df["subgroup"] == (i - k)].iloc[0, -2]] * len(
                        df[df["subgroup"] == (i - k)]
                    )
                    prev_yes = 1
                if (i + k) < max(df["subgroup"]) + 1:
                    after = [df[df["subgroup"] == (i + k)].iloc[0, -2]] * len(
                        df[df["subgroup"] == (i + k)]
                    )
                    after_yes = 1
                check_loop = True
                if (prev_yes + after_yes) == 2:
                    if mode(prev).mode[0] == mode(after).mode[0]:
                        check_loop = False

                if check_loop:
                    k = 1  # Helper
                    while len(prev) < j + 2 - i_prev_cnt:
                        k += 1

                        if (i - k) > 0:
                            prev_i = [df[df["subgroup"] == (i - k)].iloc[0, -2]] * len(
                                df[df["subgroup"] == (i - k)]
                            )
                            prev.extend(prev_i)

                        else:
                            break

                    k = 1  # Helper
                    while len(after) < j + 2 - i_prev_cnt:
                        k += 1
                        if (i + k) < max(df["subgroup"]) + 1:
                            prev_i = [df[df["subgroup"] == (i + k)].iloc[0, -2]] * len(
                                df[df["subgroup"] == (i + k)]
                            )
                            after.extend(prev_i)

                        else:
                            break
                    changeTo = prev
                    changeTo.extend(after)
                    changeTo = mode(changeTo).mode[0]
                else:
                    changeTo = mode(prev).mode[0]

                change_idx = df.index[df["subgroup"] == i].tolist()
                df.iloc[change_idx, -2] = changeTo
                df["subgroup"] = (df.iloc[:, -2] != df.iloc[:, -2].shift(1)).cumsum()
                added = (
                    df["subgroup"]
                    .value_counts()[df["subgroup"].value_counts() < (j + 1)]
                    .index.tolist()
                )
                added.sort()
                if i == i_prev:
                    i_prev_cnt += 1
                else:
                    i_prev_cnt = 0
                i_prev = i
                cnt += 1
                if cnt > max(df["subgroup"]) * (j + 2):
                    break

        # Modify the output shape so that for each task we have start frame and end frame
        output_df = pd.DataFrame(columns=["task", "startTime", "endTime"])
        for i in range(max(df["subgroup"])):
            df_i = df[df["subgroup"] == (i + 1)]
            task_str = str(class_names[int(df_i.iloc[0]["task_label"])])
            start_frame = int(min(df_i["frame"]))
            start_frame = self._frame_to_time(start_frame, video_fps)
            end_frame = int(max(df_i["frame"]))
            end_frame = self._frame_to_time(end_frame, video_fps)
            output_df = output_df.append(
                pd.DataFrame(
                    [[task_str] + [start_frame] + [end_frame]],
                    columns=["task", "startTime", "endTime"],
                )
            )

        return output_df

    def frame_to_time(self, frame, video_fps):
        """
        This function converts frame numbers into time strings: '00:00:00.00'
        """
        x = round(frame / video_fps, 2)
        x = pd.to_datetime(x, unit="s").strftime("%H:%M:%S.%f")[:-4]
        return x

    def predict(self):

        if self.video_name:
            video_name = self.video_name.rsplit('/', maxsplit=1)[-1].split('.')[0]
        else:
            video_name = self.input_video_path.as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]

        # Load class names
        classfile = open(self.model_path_classes, "r")
        class_names = [line[:-1] for line in classfile.readlines()]

        # Open video
        vid = cv2.VideoCapture(self.input_video_path.as_posix())
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")

        # Obtain FPS
        video_fps = vid.get(cv2.CAP_PROP_FPS)

        # Load Models
        intermediate_layer_model = load_model(self.model_path_densenet)
        xgbmodel = load(self.model_path_xgboost)

        cnt_list = []
        predResults = []

        # Initialize variables for the model
        hist_count = self.time_window  # take into account the previous 7 frames
        data_temporal = (
            np.zeros(hist_count * 1024).astype(int).tolist()
        )  # Start the rowlist
        data_temporal_batch = []

        # Start processing the video
        cnt = 0
        cnt1 = 0
        data = []
        while vid.isOpened():
            return_value, frame = vid.read()
            if not return_value:
                data = np.array(data, dtype="float32") / 255.0
                intermediate_test_outputs = intermediate_layer_model.predict(data)
                for intermediate_test_output in intermediate_test_outputs:
                    data_temporal = data_temporal[len(intermediate_test_output) :]
                    data_temporal.extend(intermediate_test_output)
                    data_temporal_batch.append(data_temporal)
                data_temporal_batch = np.array(data_temporal_batch)
                ypredNum = xgbmodel.predict(data_temporal_batch)
                predResults.extend(ypredNum)
                break
            image = cv2.resize(frame, (128, 128))
            image = img_to_array(image)
            data.append(image)
            cnt_list.extend([cnt])
            cnt += 1
            cnt1 += 1
            if cnt1 == self.batch_size:
                data = np.array(data, dtype="float32") / 255.0
                intermediate_test_outputs = intermediate_layer_model.predict(data)
                for intermediate_test_output in intermediate_test_outputs:
                    data_temporal = data_temporal[len(intermediate_test_output) :]
                    data_temporal.extend(intermediate_test_output)
                    data_temporal_batch.append(data_temporal)
                ypredNum = xgbmodel.predict(np.array(data_temporal_batch))
                predResults.extend(ypredNum)
                data_temporal_batch = []
                data = []
                cnt1 = 0

        vid.release()

        # Clean the output data
        out_df = pd.DataFrame({"frame": cnt_list, "task_label": predResults})
        # out_df = self.process_to_annotation_data(
        #     out_df, class_names, video_fps, self.min_len
        # )

        # Save results
        save_dir = increment_path(
            Path(__file__).joinpath(
                f'../../../../results/densenet_xgboost/{video_name}').resolve(),
            sep='_',
            mkdir=True
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir.resolve().joinpath(f'{video_name}.csv')

        out_df.to_csv(save_path)

        return save_path
