import os
import cv2
import mediapipe as mp
from PIL import Image
from tqdm import tqdm

# Input and output folders
input_root = "dataset/raw"
output_root = "dataset/cropped"
os.makedirs(output_root, exist_ok=True)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to detect and crop the first face
def detect_and_crop(image_path, save_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unreadable")

        height, width, _ = img.shape

        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height

                # Convert from relative to absolute coordinates
                x1 = max(0, int(x * width))
                y1 = max(0, int(y * height))
                x2 = min(width, int((x + w) * width))
                y2 = min(height, int((y + h) * height))

                cropped_face = img[y1:y2, x1:x2]
                if cropped_face.size == 0:
                    raise ValueError("Invalid crop dimensions")

                face_image = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)).resize((224, 224))
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


if __name__ == "__main__":
    print("[INFO] Starting face cropping using MediaPipe...")
    person_folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]

    if not person_folders:
        print("[WARNING] No folders found in dataset/raw. Exiting.")
    
    for person_id in tqdm(person_folders, desc="Processing persons"):
        input_folder = os.path.join(input_root, person_id)
        output_folder = os.path.join(output_root, person_id)
        os.makedirs(output_folder, exist_ok=True)

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"[WARNING] No images found in {input_folder}")
            continue

        for image_file in image_files:
            input_image_path = os.path.join(input_folder, image_file)
            output_image_path = os.path.join(output_folder, image_file)

            # Skip if already processed
            if os.path.exists(output_image_path):
                continue

            print(f"[INFO] Processing {input_image_path}")
            success = detect_and_crop(input_image_path, output_image_path)
            if not success:
                print(f"[FAILED] Could not crop face from {input_image_path}")
