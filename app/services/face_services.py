from fastapi import UploadFile

from app.utils.file_utils import save_upload_file


async def register_face_dummy(student_code: str, file: UploadFile):
    saved_path = await save_upload_file(file, folder="known_faces", prefix=student_code)

    return {
        "success": True,
        "studentCode": student_code,
        "savedPath": saved_path,
        "message": "Face image received successfully. Real AI registration will be added later."
    }


async def recognize_face_dummy(file: UploadFile):
    saved_path = await save_upload_file(file, folder="temp_uploads", prefix="recognize")

    return {
        "recognized": False,
        "studentCode": None,
        "confidence": 0.0,
        "savedPath": saved_path,
        "message": "Image received successfully. Real AI recognition will be added later."
    }