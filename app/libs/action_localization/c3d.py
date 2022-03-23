"""C3D model implementation."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cv2 import cv2
from tensorflow.keras.models import load_model

from app.libs.utils.general import increment_path


def c3d(
    video_path: Path,
    model_path: Path,
    video_name: str = None,
) -> Path:
    """
    C3D model implementation.

    Args:
        video_path (Path) : path to the video file,
        model_path (Path) : path to the model weights file,
        video_name (str) : optional name of the video,

    Returns:
        The path to the results .csv file.
    """
    if video_name:
        video_name = video_name.rsplit('/', maxsplit=1)[-1].split('.')[0]
    else:
        video_name = video_path.as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]

    # Load the model
    model = load_model(model_path)

    temporal_length = 16    # Defined in training (always the same)
    batch_length = 16   # In order to predict faster

    # Defnied in training (these values don't depend on the dataset)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Start processing
    vid = cv2.VideoCapture(video_path.as_posix())
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    count_temporal = 0
    count_batch = 1
    data_temporal = None
    data_batch = []
    pred_results = []
    prob_results = []

    print('start')
    start_time = time.time()

    while vid.isOpened():
        return_value, frame = vid.read()
        if not return_value:
            print('break')

            if count_batch > 1:
                pred_idxs = model.predict(x=np.array(data_batch))
                prob_results.extend(np.amax(pred_idxs, axis=1))

                # Obtain class results from predictions
                if len(pred_idxs[0]) == 1:
                    pred_idxs = np.round(pred_idxs).astype(int).flatten()
                else:
                    pred_idxs = np.argmax(pred_idxs, axis=1)
                pred_results.extend(pred_idxs)
            break

        image = cv2.resize(frame, (112, 112))
        image = image/255
        image = (image - mean) / std

        if count_temporal < temporal_length:
            if count_temporal == 0:
                data_temporal = np.array([image])
            else:
                data_temporal = np.append(data_temporal, [image], axis=0)

        if count_temporal >= temporal_length:
            data_temporal[:-1] = data_temporal[1:]
            data_temporal[-1] = image

        if count_temporal >= (temporal_length - 1):
            if count_batch < batch_length:
                if count_batch == 1:
                    data_batch = np.array([data_temporal])
                else:
                    data_batch = np.append(data_batch, [data_temporal], axis=0)
                count_batch += 1
            else:
                data_batch = np.append(data_batch, [data_temporal], axis=0)
                pred_idxs = model.predict(x=data_batch)
                prob_results.extend(np.amax(pred_idxs, axis=1))

                # Obtain class results from predictions
                if len(pred_idxs[0]) == 1:
                    pred_idxs = np.round(pred_idxs).astype(int).flatten()
                else:
                    pred_idxs = np.argmax(pred_idxs, axis=1)
                pred_results.extend(pred_idxs)
                count_batch = 1
                data_batch = []

                print('count: ', count_temporal)
        count_temporal += 1

    end_time = time.time()
    print(f'total_time: {end_time - start_time:0.3f} sec')

    # Save results
    save_dir = increment_path(
        Path(__file__).joinpath(
            f'../../../../results/C3D/{video_name}').resolve(),
        sep='_',
        mkdir=True
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir.resolve().joinpath(f'{video_name}.csv')

    # Add task_name column from classes file/upload info here, if needed
    df = pd.DataFrame({'task': pred_results, 'prob': prob_results})
    df.to_csv(save_path)

    vid.release()

    return save_path


if __name__ == '__main__':
    test_video = Path(__file__).joinpath(
        '../../../../test_videos/v-scan_papers_computer-clip2-1.mp4').resolve()

    test_model = Path(__file__).joinpath(
        '../../../../model_weights/c3d/C3D_scratch_best_optimo_v1.1.hdf5').resolve()

    if test_video.exists() and test_model.exists():
        test_save = c3d(
            video_path=test_video,
            model_path=test_model,
            video_name=test_video.as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0],
        )
        print(test_save)
    else:
        print("Test video: 'v-scan_papers_computer-clip2-1.mp4' ",
              "or test model: 'C3D_scratch_best_optimo_v1.1.hdf5' ",
              "don't exist. Please check files.")
