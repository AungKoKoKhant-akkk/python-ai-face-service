from fastapi import UploadFile
from pathlib import Path
import shutil
import json
import os


MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"


async def register_face(student_code: str, file: UploadFile):
    """
    Register one student's face.

    Steps:
    1. Save uploaded image temporarily
    2. Use DeepFace to detect and generate embedding
    3. Check exactly one face exists
    4. Save image and embedding under known_faces/{student_code}
    """

    if file is None:
        return {
            "success": False,
            "studentCode": student_code,
            "message": "Face image file is required"
        }

    if not file.content_type or not file.content_type.startswith("image/"):
        return {
            "success": False,
            "studentCode": student_code,
            "message": "Only image files are allowed"
        }

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_file_path = temp_dir / f"{student_code}_register_temp.jpg"

    try:
        # Save uploaded file temporarily
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Import DeepFace lazily so the app can start without heavy ML deps
        try:
            from deepface import DeepFace
        except Exception as e:
            return {
                "success": False,
                "studentCode": student_code,
                "message": (
                    "DeepFace is not available: "
                    f"{e}. Install 'deepface' and required packages (e.g. 'tf-keras')"
                )
            }

        # DeepFace generates embedding.
        # enforce_detection=True means it throws error if no face is detected.
        embedding_result = DeepFace.represent(
            img_path=str(temp_file_path),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )

        # DeepFace.represent returns a list of detected faces.
        if len(embedding_result) == 0:
            return {
                "success": False,
                "studentCode": student_code,
                "message": "No face detected in image"
            }

        if len(embedding_result) > 1:
            return {
                "success": False,
                "studentCode": student_code,
                "message": "Multiple faces detected. Please upload only one student's face."
            }

        face_data = embedding_result[0]
        embedding = face_data["embedding"]
        face_confidence = face_data.get("face_confidence", None)
        facial_area = face_data.get("facial_area", None)

        student_face_dir = Path("known_faces") / student_code
        student_face_dir.mkdir(parents=True, exist_ok=True)

        final_image_path = student_face_dir / "face.jpg"
        final_embedding_path = student_face_dir / "embedding.json"

        # Save final face image
        shutil.copyfile(temp_file_path, final_image_path)

        # Save embedding data
        embedding_json = {
            "studentCode": student_code,
            "modelName": MODEL_NAME,
            "detectorBackend": DETECTOR_BACKEND,
            "embedding": embedding,
            "faceConfidence": face_confidence,
            "facialArea": facial_area
        }

        with final_embedding_path.open("w", encoding="utf-8") as f:
            json.dump(embedding_json, f)

        return {
            "success": True,
            "studentCode": student_code,
            "imagePath": str(final_image_path),
            "embeddingPath": str(final_embedding_path),
            "faceConfidence": face_confidence,
            "message": "Face registered successfully"
        }

    except ValueError:
        return {
            "success": False,
            "studentCode": student_code,
            "message": "No face detected. Please upload a clear face image."
        }

    except Exception as e:
        return {
            "success": False,
            "studentCode": student_code,
            "message": f"Face registration failed: {str(e)}"
        }

    finally:
        if temp_file_path.exists():
            os.remove(temp_file_path)


async def recognize_face_dummy(file: UploadFile):
    return {
        "recognized": False,
        "studentCode": None,
        "confidence": 0.0,
        "message": "Real face recognition will be added in Step 13."
    }