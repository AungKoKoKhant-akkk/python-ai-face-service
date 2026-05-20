from fastapi import APIRouter, UploadFile, File, Form
from app.services.face_services import register_face_dummy, recognize_face_dummy
router = APIRouter()

@router.post("/register-face")
async def register_face(
    studentCode: str = Form(...),
    file: UploadFile = File(...)
):
    return await register_face_dummy(studentCode, file)


@router.post("/recognize-face")
async def recognize_face(
    file: UploadFile = File(...)
):
    return await recognize_face_dummy(file)