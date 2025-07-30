#batch_crop_faces.py

import os
from pathlib import Path
from src.face_preprocess import detect_and_crop_face

def batch_crop_faces(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    for person_dir in input_root.iterdir():
        if not person_dir.is_dir():
            continue

        output_person_dir = output_root / person_dir.name
        output_person_dir.mkdir(parents=True, exist_ok=True)

        for img_file in person_dir.glob("*.jpg"):
            output_img_path = output_person_dir / img_file.name
            face = detect_and_crop_face(
                str(img_file),
                str(output_img_path),
                save=True
            )

            if face is None:
                print(f"⚠️ No face found in {img_file.name}")

if __name__ == "__main__":
    batch_crop_faces("images/criminals", "images/cropped_criminals")
