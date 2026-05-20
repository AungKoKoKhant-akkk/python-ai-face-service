from fastapi import APIRouter, UploadFile, File, Form
from app.services.face_service import register_face, recognize_face

router = APIRouter()


@router.post("/register-face")
async def register_student_face(
    studentCode: str = Form(...),
    file: UploadFile = File(...)
):
    return await register_face(studentCode, file)


@router.post("/recognize-face")
async def recognize_student_face(
    file: UploadFile = File(...)
):
    return await recognize_face(file)