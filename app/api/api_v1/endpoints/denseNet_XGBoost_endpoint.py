import os
import glob
from pathlib import Path
from typing import Any, Optional, List
import aiofiles

from fastapi import APIRouter, File, UploadFile, HTTPException, status

# from app import schemas
from app.api import deps
from app.libs.action_localization.denseNet_XGBoost import DenseNet_XGBoost

router = APIRouter()


@router.post("/uploadweightsdensenetXGB/")
async def densenet_upload_weights(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload densenet weights.
    """
    if (
        file.content_type not in ['application/octet-stream']
        or file.filename.split('.')[-1] not in ['h5']
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload only .h5 files"
        )

    async with aiofiles.open(
        f'model_weights/densenet_xgboost/densenet/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()  # async read upload
        await out_file.write(model_data)  # async write to local file

    return {"filename": file.filename}


@router.post("/uploadweightsDNxgboost/")
async def xgboost_upload_weights(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload xgboost weights.
    """
    if (
        file.content_type not in ['application/octet-stream']
        or file.filename.split('.')[-1] not in ['joblib']
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload only .joblib files"
        )

    async with aiofiles.open(
        f'model_weights/densenet_xgboost/xgboost/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()  # async read upload
        await out_file.write(model_data)  # async write to local file

    return {"filename": file.filename}


@router.post("/uploadclasses/")
async def upload_classes(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload class names.
    """
    if (
        file.content_type not in ['text/plain']
        or file.filename.split('.')[-1] not in ['txt']
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload only .txt files"
        )

    async with aiofiles.open(
        f'model_weights/densenet_xgboost/classes/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()  # async read upload
        await out_file.write(model_data)  # async write to local file

    return {"filename": file.filename}


@router.get("/weights/")
async def densenet_xgboost_weights() -> List[str]:
    """
    Get available weights options for model.
    """
    return {
        'denseNet': [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
                     for f in glob.glob("model_weights/densenet_xgboost/densenet/*.h5")],
        'XGBoost': [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
                    for f in glob.glob("model_weights/densenet_xgboost/xgboost/*.joblib")],
        'classNames': [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
                       for f in glob.glob("model_weights/densenet_xgboost/classes/*.txt")]
    }


@router.post("/actiondetection/")
async def densenet_xgboost_action_localization(
    files: List[UploadFile] = File(...),
    weights_densenet: Optional[str] = "denseXgB_model_mylayer",
    weights_xgboost: Optional[str] = "recognition_xgboost_prev_frames",
    classNames: Optional[str] = "classes",
    save_upload_to_file: bool = False,
) -> Any:
    """
    Get densenet_xgboost action localization result for the video file.
    """
    # Obtain the model paths
    model_path_densenet = Path(f"model_weights/densenet_xgboost/densenet/{weights_densenet}.h5")
    model_path_xgboost = Path(f"model_weights/densenet_xgboost/xgboost/{weights_xgboost}.joblib")
    model_path_classes = Path(f"model_weights/densenet_xgboost/classes/{classNames}.txt")

    if (
        not os.path.isfile(model_path_densenet)
        or not os.path.isfile(model_path_xgboost)
        or not os.path.isfile(model_path_classes)
    ):
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

            model = DenseNet_XGBoost(
                input_video_path=video_path,
                model_path_densenet=model_path_densenet,
                model_path_xgboost=model_path_xgboost,
                model_path_classes=model_path_classes,
                video_name=file.filename,
            )
            save_path = model.predict()
            print(video_path)

        finally:
            if not save_upload_to_file:
                Path.unlink(video_path)  # Delete the temp file

    return {
        "model_weights_rgb": weights_densenet,
        "model_weights_of": weights_xgboost,
        "classNames": classNames,
        "results_path": save_path
    }
