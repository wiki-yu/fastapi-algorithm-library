import os
from pathlib import Path
from typing import Any, Optional, List
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import RedirectResponse

from app.api import deps
from app.libs.detection_tracking.deep_sort_pytorch.track import detect

router = APIRouter()


@router.post("/uploadweights")
async def redirect_yolov5_upload_weights(
    file: UploadFile = File(...)
) -> Any:
    """Upload YOLOv5 weights."""
    response = RedirectResponse(url='../yolov5/uploadweights/')
    return response


@router.get("/yolov5weights")
async def redirect_yolov5_weights():
    """Get available weights options for yolov5 model."""
    response = RedirectResponse(url='../yolov5/weights/')
    return response


@router.post("/videotracking")
def deepsort_video_tracking(
    files: List[UploadFile] = File(...),
    weights: Optional[str] = "best",
    save_upload_to_file: bool = False,
) -> Any:
    """
    Get DeepSORT tracking result for video file.
    """
    # Set model params
    model_path = Path(f"model_weights/yolov5/{weights}.pt")
    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model weights not available."
        )

    outputs = []
    for file in files:
        try:
            if file.content_type in ['video/mp4']:
                if save_upload_to_file:
                    video_path = Path(f'uploads/video/{file.filename}')
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    deps.save_upload_file(upload_file=file, destination=video_path)
                else:
                    # Upload is still written to file, but deleted after use
                    video_path = deps.save_upload_file_tmp(upload_file=file)
            else:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Please upload only .mp4 files."
                )

            detections = detect(video_path.as_posix(), model_path)
            outputs = [i.tolist() for i in detections]

        finally:
            if not save_upload_to_file:
                Path.unlink(video_path)  # Delete the temp file

    return {
        "model weights": weights,
        # Discussion about the output format
        "results": outputs
    }
