import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Generator

from fastapi import UploadFile
from app.db.session import SessionLocal


def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        # upload_file.file.seek(0)
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        # upload_file.file.seek(0)
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:  # do we want delete=True?
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
        tmp.close()
    return tmp_path


def handle_upload_file(
    upload_file: UploadFile, handler: Callable[[Path], None]
) -> None:
    tmp_path = save_upload_file_tmp(upload_file)
    try:
        handler(tmp_path)  # Do something with the saved temp file
    finally:
        tmp_path.unlink()  # Delete the temp file


# Database dependency
def get_db() -> Generator:
    '''
    A new SQLAlchemy SessionLocal that will be used in a single
    request, and then close it once the request is finished
    '''
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()
