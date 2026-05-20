from fastapi import UploadFile
from deepface import DeepFace
from pathlib import Path
import shutil
import json
import os
import numpy as np


MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"

# Smaller distance means more similar.
# You can adjust later if recognition is too strict or too loose.
RECOGNITION_THRESHOLD = 0.40


async def register_face(student_code: str, file: UploadFile):
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
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        embedding_result = DeepFace.represent(
            img_path=str(temp_file_path),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )

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

        shutil.copyfile(temp_file_path, final_image_path)

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


async def recognize_face(file: UploadFile):
    if file is None:
        return {
            "recognized": False,
            "studentCode": None,
            "confidence": 0.0,
            "message": "Face image file is required"
        }

    if not file.content_type or not file.content_type.startswith("image/"):
        return {
            "recognized": False,
            "studentCode": None,
            "confidence": 0.0,
            "message": "Only image files are allowed"
        }

    known_faces_dir = Path("known_faces")

    if not known_faces_dir.exists():
        return {
            "recognized": False,
            "studentCode": None,
            "confidence": 0.0,
            "message": "No registered faces found"
        }

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_file_path = temp_dir / "recognize_temp.jpg"

    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        embedding_result = DeepFace.represent(
            img_path=str(temp_file_path),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )

        if len(embedding_result) == 0:
            return {
                "recognized": False,
                "studentCode": None,
                "confidence": 0.0,
                "message": "No face detected in image"
            }

        if len(embedding_result) > 1:
            return {
                "recognized": False,
                "studentCode": None,
                "confidence": 0.0,
                "message": "Multiple faces detected. Version 1 supports one face only."
            }

        unknown_embedding = embedding_result[0]["embedding"]

        best_match_student_code = None
        best_distance = None

        for student_dir in known_faces_dir.iterdir():
            if not student_dir.is_dir():
                continue

            embedding_file = student_dir / "embedding.json"

            if not embedding_file.exists():
                continue

            with embedding_file.open("r", encoding="utf-8") as f:
                saved_data = json.load(f)

            known_embedding = saved_data["embedding"]
            student_code = saved_data["studentCode"]

            distance = cosine_distance(unknown_embedding, known_embedding)

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_match_student_code = student_code

        if best_match_student_code is None:
            return {
                "recognized": False,
                "studentCode": None,
                "confidence": 0.0,
                "message": "No registered student embeddings found"
            }

        confidence = calculate_confidence(best_distance)

        if best_distance <= RECOGNITION_THRESHOLD:
            return {
                "recognized": True,
                "studentCode": best_match_student_code,
                "confidence": confidence,
                "distance": best_distance,
                "message": "Face recognized successfully"
            }

        return {
            "recognized": False,
            "studentCode": None,
            "confidence": confidence,
            "distance": best_distance,
            "bestMatchStudentCode": best_match_student_code,
            "message": "Face not recognized. Distance is higher than threshold."
        }

    except ValueError:
        return {
            "recognized": False,
            "studentCode": None,
            "confidence": 0.0,
            "message": "No face detected. Please upload a clear face image."
        }

    except Exception as e:
        return {
            "recognized": False,
            "studentCode": None,
            "confidence": 0.0,
            "message": f"Face recognition failed: {str(e)}"
        }

    finally:
        if temp_file_path.exists():
            os.remove(temp_file_path)


def cosine_distance(embedding1, embedding2):
    vector1 = np.array(embedding1)
    vector2 = np.array(embedding2)

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 1.0

    similarity = dot_product / (norm1 * norm2)

    distance = 1 - similarity

    return float(distance)


def calculate_confidence(distance):
    confidence = max(0.0, 1.0 - distance)
    return round(confidence, 4)