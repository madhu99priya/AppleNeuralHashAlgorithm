import os
import cv2
from PIL import Image
from tqdm import tqdm
import mediapipe as mp
import time

# Input and output folders
input_root = "dataset1/raw"
output_root = "dataset1/cropped"
os.makedirs(output_root, exist_ok=True)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Function to detect and crop face using MediaPipe
def detect_and_crop(image_path, save_path):
    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[WARN] Unable to read image: {image_path}")
            return False

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
            results = face_detector.process(image_rgb)

            if not results.detections:
                print(f"[INFO] No face detected in: {image_path}")
                return False

            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            h, w, _ = image_rgb.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            x, y = max(0, x), max(0, y)
            x2 = min(w, x + width)
            y2 = min(h, y + height)

            cropped_face = image_rgb[y:y2, x:x2]
            if cropped_face.size == 0:
                print(f"[WARN] Empty crop for image: {image_path}")
                return False

            face_image = Image.fromarray(cropped_face).resize((224, 224))
            face_image.save(save_path)
            return True

    except Exception as e:
        print(f"[ERROR] Failed processing {image_path}: {e}")
        return False

# Measure start time
start_time = time.time()

# Traverse all person folders
person_folders = [
    f for f in os.listdir(input_root)
    if os.path.isdir(os.path.join(input_root, f))
]

for person_id in tqdm(person_folders, desc="Processing persons"):
    input_folder = os.path.join(input_root, person_id)
    output_folder = os.path.join(output_root, person_id)
    os.makedirs(output_folder, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)

        if os.path.exists(output_image_path):
            continue

        detect_and_crop(input_image_path, output_image_path)

# Measure end time and print duration
end_time = time.time()
elapsed = end_time - start_time
print(f"\n✅ Face cropping completed in {elapsed:.2f} seconds.")
