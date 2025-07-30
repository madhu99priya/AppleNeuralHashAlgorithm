#face_preprocess.py

import os
import cv2
import face_recognition
from pathlib import Path

def detect_and_crop_face(image_path, output_path=None, save=True):
    
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        print(f"No face found in {image_path}")
        return None

    # Find the largest face (by bounding box area)
    largest_face = max(face_locations, key=lambda box: (box[2] - box[0]) * (box[1] - box[3]))
    top, right, bottom, left = largest_face
    
    face_image = image[top:bottom, left:right]

    # Resize face to standard size (e.g., 160x160)
    face_image = cv2.resize(face_image, (160, 160))

    if save and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

    return face_image