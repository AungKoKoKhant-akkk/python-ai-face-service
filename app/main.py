

from fastapi import FastAPI
from app.routes.face_routes import router as face_router

app = FastAPI(
    title = "Attendance Face Recognition AI Service",
    description = "Python FastAPI service for student face registration and recognition",
    version = "1.0",
)

app.include_router(face_router, prefix="/ai", tags=["Face Recognition"])

@app.get("/")
def root():
    return {
        "message": "Attendance AI Service is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "service": "python-ai-service"
    }
