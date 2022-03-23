import os
import glob
import aiofiles
from pathlib import Path
from typing import Any, Optional, List
from fastapi import APIRouter, File, UploadFile, HTTPException, status

from app.api import deps
from app.libs.action_localization.spatio_temporal_pytorch.spatio_temporal import Spatio_Temporal_Model

router = APIRouter()


@router.post("/uploadweightsSpatioTemporal/")
async def spatio_temporal_upload_weights(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload spatio_temporal model weights

    Args:
        file (UploadFile) : a spatio-temporal model weights file (.pth )

    Returns:
        The filename that was successfully uploaded
    """
    if (
        file.content_type not in ['application/octet-stream']
        or file.filename.rsplit('.', maxsplit=1)[-1] not in ['pth']
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload only .pth files"
        )

    async with aiofiles.open(
        f'model_weights/spatio_temporal/action_localization/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()
        await out_file.write(model_data)

    return {"filename": file.filename}


@router.post("/uploadweightsBackbone2D/")
async def backbone2d_upload_weights(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload 2D backbone model weights

    Args:
        file (UploadFile) : a YOLO weights file

    Returns:
        The filename that was successfully uploaded
    """
    if (
        file.content_type not in ['application/octet-stream']
        or file.filename.rsplit('.', maxsplit=1)[-1] not in ['weights']
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload only .weights files"
        )

    async with aiofiles.open(
        f'model_weights/spatio_temporal/backbone2d/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()
        await out_file.write(model_data)

    return {"filename": file.filename}


@router.post("/uploadweightsBackbone3d/")
async def backbone3d_upload_weights(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload 3d backbone model weights.

    Args:
        file (UploadFile) : a resnext101 weights file (.pth)

    Returns:
        The filename that was successfully uploaded.
    """
    if (
        file.content_type not in ['application/octet-stream']
        or file.filename.rsplit('.', maxsplit=1)[-1] not in ['pth']
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload only .pth files"
        )

    async with aiofiles.open(
        f'model_weights/spatio_temporal/backbone3d/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()
        await out_file.write(model_data)

    return {"filename": file.filename}


@router.get("/weights/")
async def spatio_temporal_weights() -> List[str]:
    """
    Get available weights options for model.

    Returns:
        List[str] of all the 3 model weights files currently available to use.
    """
    return {
            'spatio_temporal_model': [
                Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
                for f in glob.glob("model_weights/spatio_temporal/action_localization/*.pth")
            ],
            '3D backbone': [
                Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
                for f in glob.glob("model_weights/spatio_temporal/backbone3d/*.pth")
            ],
            '2D backbone': [
                Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
                for f in glob.glob("model_weights/spatio_temporal/backbone2d/*.weights")
            ]
    }


@router.post("/actiondetection/")
def spatio_temporal_action_localization(
    files: List[UploadFile] = File(...),
    weights_spatio_temporal: Optional[str] = "yowo_ucsp_16f_best",
    weights_backbone_2d: Optional[str] = "yolo",
    weights_backbone_3d: Optional[str] = "resnext-101-kinetics",
    save_upload_to_file: bool = False,
) -> Any:
    """
    Get Spatio-temporal action localization result for the video file.

    Args:
        files (List[UploadFile]) : a list of .mp4 files
        weights_spatio_temporal (str) : name of the spatio-temporal model weights file (.pth)
        weights_backbone_2d (str) : name of the 2D backbone weights file (.weights)
        weights_backbone_3d (str) : name of the 3D backbone weights file (.pth)
        save_upload_to_file (bool) : save upload video, or delete after processing

    Returns:
        JSON format:
            model_weights: name of spatio-temporal model weights used
            results_path: path at which results were saved
    """

    # Obtain the model path
    model_path_spatio_temporal = Path(
        f"model_weights/spatio_temporal/action_localization/{weights_spatio_temporal}.pth"
    )
    model_path_backbone2d = Path(f"model_weights/spatio_temporal/backbone2d/{weights_backbone_2d}.weights")
    model_path_backbone3d = Path(f"model_weights/spatio_temporal/backbone3d/{weights_backbone_3d}.pth")

    if (
        not os.path.isfile(model_path_spatio_temporal)
        or not os.path.isfile(model_path_backbone2d)
        or not os.path.isfile(model_path_backbone3d)
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model weights not available."
        )

    for file in files:
        try:
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

            model = Spatio_Temporal_Model(
                video_path=video_path,
                model_path_spatio_temporal=model_path_spatio_temporal,
                model_path_backbone2d=model_path_backbone2d,
                model_path_backbone3d=model_path_backbone3d,
                video_name=file.filename
            )
            save_path = model.action_detection(verbose=True)

        finally:
            if not save_upload_to_file:
                Path.unlink(video_path)

    return {
        "amodel_weights": weights_spatio_temporal,
        "results_path": save_path
    }
