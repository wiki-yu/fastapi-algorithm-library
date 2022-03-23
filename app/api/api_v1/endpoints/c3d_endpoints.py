import os
import glob
from pathlib import Path
from typing import Any, Optional, List
import aiofiles

from fastapi import APIRouter, File, UploadFile, HTTPException, status

# from app import schemas
from app.api import deps
from app.libs.action_localization.c3d import c3d


router = APIRouter()


@router.post("/uploadweights/")
async def c3d_upload_weights(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload C3D weights.

    Args:
        file (UploadFile) : a C3D weights binary .hdf5 file

    Returns:
        The filename that was successfully uploaded.
    """
    if (
        file.content_type not in ['application/octet-stream']
        or file.filename.rsplit('.', maxsplit=1)[-1] not in ['hdf5']
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload only .hdf5 files"
        )

    async with aiofiles.open(
        f'model_weights/c3d/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()  # async read upload
        await out_file.write(model_data)  # async write to local file

    return {"filename": file.filename}


@router.get("/weights/")
async def c3d_weights() -> List[str]:
    """
    Get available weights options for model.

    Returns:
        List[str] of all C3D model weights files currently available to use.
    """
    return [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
            for f in glob.glob("model_weights/c3d/*.hdf5")]


@router.post("/actiondetection/")
def c3d_action_localization(
    files: List[UploadFile] = File(...),
    weights: Optional[str] = "C3D",
    save_upload_to_file: bool = False,
) -> Any:
    """
    Get C3D action localization result for the video file.

    Args:
        files (List[UploadFile]) : a list of .mp4 files
        weights (str) : name of a C3D weights binary .hdf5 file
        save_upload_to_file (bool) : save upload video, or delete after processing

    Returns:
        JSON format:
            model_weights: name of weights used
            results_path: path at which results were saved
    """
    # Obtain the model path
    model_path = Path(f"model_weights/c3d/{weights}.hdf5")
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

            save_path = c3d(
                video_path=video_path,
                model_path=model_path,
                video_name=file.filename
            )

        finally:
            if not save_upload_to_file:
                Path.unlink(video_path)  # Delete the temp file

    return {
        "model_weights": weights,
        # Discussion about the output format
        # "results": df.values.tolist() # 32,000 frames, too long for response
        "results_path": save_path
        # 'status': 'Complete'
    }
