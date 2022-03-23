import os
import glob
from pathlib import Path
from typing import Any, Optional, List
import aiofiles

from fastapi import APIRouter, File, UploadFile, HTTPException, status

# from app import schemas
from app.api import deps
from app.libs.action_localization.rolling_average import Vgg_RollAverage_RealTime

router = APIRouter()


@router.post("/uploadweights/")
async def rolling_average_upload_weights(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload rolling_average weights.
    """
    if (
        file.content_type not in ['application/octet-stream']
        or file.filename.split('.')[-1] not in ['hdf5']
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload only .hdf5 files"
        )

    async with aiofiles.open(
        f'model_weights/rolling_average/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()  # async read upload
        await out_file.write(model_data)  # async write to local file

    return {"filename": file.filename}


@router.get("/weights/")
async def rolling_average_weights() -> List[str]:
    """
    Get available weights options for model.
    """
    return [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
            for f in glob.glob("model_weights/rolling_average/*.hdf5")]


@router.post("/actiondetection/")
async def rolling_average_action_localization(
    files: List[UploadFile] = File(...),
    weights: Optional[str] = "vgg_tf_vgg_test_77",
    save_upload_to_file: bool = False,
    rollavg: Optional[bool] = True,
) -> Any:
    """
    Get roll_average action localization result for the video file.
    """
    # Obtain the model path
    model_path = Path(f"model_weights/rolling_average/{weights}.hdf5")
    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model weights not available."
        )

    for file in files:
        try:
            # Obtain the video path
            if file.content_type in ['video/mp4']:
                if save_upload_to_file:
                    video_path = Path(f'uploads/video/{file.filename}')
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    deps.save_upload_file(upload_file=file, destination=video_path)
                else:
                    video_path = deps.save_upload_file_tmp(upload_file=file)
            else:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Please upload only .mp4 files."
                )

            data_generator_object = Vgg_RollAverage_RealTime()
            save_path = data_generator_object.real_time_prediction(
                video_path=video_path,
                weights_filepath=model_path,
                video_name=file.filename,
                roll_average=rollavg
            )

        finally:
            if not save_upload_to_file:
                Path.unlink(video_path)  # Delete the temp file

    return {
        "model_weights": weights,
        "results_path": save_path
    }
