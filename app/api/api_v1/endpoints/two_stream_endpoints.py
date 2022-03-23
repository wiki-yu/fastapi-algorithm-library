import os
import glob
from pathlib import Path
from typing import Any, Optional, List
import aiofiles

from fastapi import APIRouter, File, UploadFile, HTTPException, status

# from app import schemas
from app.api import deps
from app.libs.action_localization.two_stream import Two_Stream_RealTime

router = APIRouter()


@router.post("/uploadweightsrgb/")
async def two_stream_upload_weights_rgb(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload two_stream weights.
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
        f'model_weights/two_stream/rgb/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()  # async read upload
        await out_file.write(model_data)  # async write to local file

    return {"filename": file.filename}


@router.post("/uploadweightsof/")
async def two_stream_upload_weights_of(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload two_stream weights.
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
        f'model_weights/two_stream/of/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()  # async read upload
        await out_file.write(model_data)  # async write to local file

    return {"filename": file.filename}


@router.get("/weights/")
async def two_stream_weights() -> List[str]:
    """
    Get available weights options for model.
    """
    return {
        'rgb': [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
                for f in glob.glob("model_weights/two_stream/rgb/*.hdf5")],
        'of': [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
               for f in glob.glob("model_weights/two_stream/of/*.hdf5")]
    }


@router.post("/actiondetection/")
async def two_stream_action_localization(
    files: List[UploadFile] = File(...),
    weights_rgb: Optional[str] = "vgg_tf_vgg_test_77",
    weights_of: Optional[str] = "vgg_test_of",
    save_upload_to_file: bool = False,
    rollavg: bool = True,
) -> Any:
    """
    Get two_stream action localization result for the video file.
    """
    # Obtain the model paths
    model_path_rgb = Path(f"model_weights/two_stream/rgb/{weights_rgb}.hdf5")
    model_path_of = Path(f"model_weights/two_stream/of/{weights_of}.hdf5")
    if not os.path.isfile(model_path_rgb) or not os.path.isfile(model_path_of):
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

            data_gen_object = Two_Stream_RealTime(size=224, roll_len=16)
            save_path = data_gen_object.real_time_prediction(
                video_path=video_path,
                weight_path_rgb=model_path_rgb,
                weight_path_of=model_path_of,
                video_name=file.filename,
                roll_average=rollavg
            )

        finally:
            if not save_upload_to_file:
                Path.unlink(video_path)  # Delete the temp file

    return {
        "model_weights_rgb": weights_rgb,
        "model_weights_of": weights_of,
        "results_path": save_path
    }
