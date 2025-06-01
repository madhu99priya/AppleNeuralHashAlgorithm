import os
import cv2
from mtcnn import MTCNN
from PIL import Image
from tqdm import tqdm

# Input and output folders
input_root = "dataset/raw"
output_root = "dataset/cdropped"
os.makedirs(output_root, exist_ok=True)

# Initialize MTCNN once
detector = MTCNN()

# Function to detect and crop the first face
def detect_and_crop(image_path, save_path):
    try:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img)
        if faces:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            cropped_face = img[y:y+h, x:x+w]
            face_image = Image.fromarray(cropped_face).resize((224, 224))
            face_image.save(save_path)
            return True
    except Exception as e:
        print(f"[ERROR] Failed processing {image_path}: {e}")
    return False

# Traverse all person folders
person_folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]

for person_id in tqdm(person_folders, desc="Processing persons"):
    input_folder = os.path.join(input_root, person_id)
    output_folder = os.path.join(output_root, person_id)
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)

        # Skip if already processed
        if os.path.exists(output_image_path):
            continue

        detect_and_crop(input_image_path, output_image_path)
