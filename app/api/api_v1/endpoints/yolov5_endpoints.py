import os
import glob
from pathlib import Path
import aiofiles

from typing import Any, Optional, List
from pandas.core.frame import DataFrame

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse

from app.libs.detection_tracking.yolov5 import detect

from app import schemas
from app.api import deps

router = APIRouter()


@router.post("/uploadweights", responses=schemas.yolov5uploadweights_response_examples)
async def yolov5_upload_weights(
    file: UploadFile = File(...)
) -> Any:
    """
    Upload YOLOv5 weights.

    Args:
        file (UploadFile) : a YOLOv5 weights binary .pt file

    Returns:
        The filename that was successfully uploaded.
    """
    if (
        file.content_type not in ['application/octet-stream']
        or file.filename.rsplit('.', maxsplit=1)[-1] not in ['pt']
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload only .pt files."
        )

    async with aiofiles.open(
        f'model_weights/yolov5/{file.filename}', 'wb'
    ) as out_file:
        model_data = await file.read()  # async read upload
        await out_file.write(model_data)  # async write to local file

    return {"filename": file.filename}


@router.get("/weights", responses=schemas.yolov5getweights_response_examples)
async def yolov5_weights() -> List[str]:
    """
    Get available weights options for model.

    Returns:
        List[str] of all YOLOv5 model weights files currently available to use.
    """
    return [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
            for f in glob.glob("model_weights/yolov5/*.pt")]


@router.post(
    "/detect",
    response_model=schemas.Yolov5Response,
    responses=schemas.yolov5_response_examples
)
def yolov5_detect(
    files: List[UploadFile] = File(...),
    weights: Optional[str] = "yolov5s",
    save_upload_to_file: bool = False,
    detection_format: Optional[str] = "xyxy"
) -> Any:
    """
    Get YOLOv5 detection result for image and video files.

    Args:
        files (List[UploadFile]) : a list of .jpeg, .png, or .mp4 files
        weights (str) : name of a YOLOv5 weights binary .pt file
        save_upload_to_file (bool) : whether to save upload video to file or use from memory
        detection_format (str) : the format detections are made, i.e. ["xyxy", "xyxyn", "xywh", "xywhn"]

    Returns:
        JSON format:
            model_weights: name of weights used
            filenames: list of filenames uploaded
            results_format: detection format selected
            results: list of results
            results_by_detection: list of results by detection
    """
    if detection_format not in ["xyxy", "xyxyn", "xywh", "xywhn"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="detection_format must be 'xyxy', 'xyxyn', 'xywh', or 'xywhn'."
        )

    # Set model params
    model_path = f"model_weights/yolov5/{weights}.pt"
    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model weights not available."
        )

    results_list = []
    results_by_detection_list = []
    for file in files:
        if file.content_type not in ['image/jpeg', 'image/png', 'video/mp4']:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Please upload only .jpeg, .png, or .mp4 files."
            )

        try:
            if save_upload_to_file:
                video_path = Path(
                    f'uploads/{file.content_type.split("/", maxsplit=1)[0]}/{file.filename}'
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
                deps.save_upload_file(upload_file=file, destination=video_path)
            else:
                # Upload is still written to file, but deleted after use
                video_path = deps.save_upload_file_tmp(upload_file=file)

            results_dict = detect.run(
                source=video_path.as_posix(),
                weights=model_path,
                conf_thres=0.25,
                imgsz=[640],
                project='results/yolov5/detect',
                name=f'{os.path.splitext(file.filename)[0]}',
                sep='_',
                save_txt=True, save_conf=True,  # cls, *xywh, conf
                save_crop=False,
                line_thickness=2,
            )

            results_df: DataFrame = results_dict[detection_format]

            results_list.append(results_df.to_dict())
            results_by_detection_list.append(results_df.transpose().to_dict())

        finally:
            if not save_upload_to_file:
                Path.unlink(video_path)  # Delete the temp file

    return {
        "model_weights": weights,
        "filenames": [file.filename for file in files],
        "results_format": detection_format,
        "results": results_list,
        "results_by_detection": results_by_detection_list
    }


@router.get("/")
async def main():
    content = """
    <body>
        <form action="/api/yolov5/detect/" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit">
        </form>
    </body>
    """
    return HTMLResponse(content=content)
