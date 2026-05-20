from pathlib import Path
import time
from fastapi import UploadFile


async def save_upload_file(file: UploadFile, folder: str, prefix: str) -> str:
    """Save an UploadFile to disk and return the saved path.

    Produces a filename using the given prefix and a millisecond timestamp
    and preserves the original file extension (defaults to .jpg).
    """
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    original_name = file.filename or "upload.jpg"
    extension = Path(original_name).suffix or ".jpg"

    timestamp_ms = int(time.time() * 1000)
    file_name = f"{prefix}_{timestamp_ms}{extension}"
    file_path = folder_path / file_name

    # Read uploaded file contents (UploadFile is async) and write to disk
    contents = await file.read()
    file_path.write_bytes(contents)

    return str(file_path)