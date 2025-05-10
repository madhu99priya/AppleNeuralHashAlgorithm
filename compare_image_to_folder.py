import os
import sys
import cv2
from preprocessing_mediapipe import NeuralHash, Hamming

# --- Config ---
parent_folder = "./dataset"  
file_formats = (".jpg", ".png")
threshold = 10
split_char = "_"

# --- Command-line input ---
if len(sys.argv) != 2:
    print("❌ Usage: python compare_image_to_nested_folders.py <input_image_path>")
    sys.exit(1)

input_image = sys.argv[1]
if not os.path.isfile(input_image):
    print(f"❌ Error: File not found - {input_image}")
    sys.exit(1)

# --- Initialize NeuralHash and Hamming ---
nh = NeuralHash()
hamming = Hamming(split_char=split_char, output_format=1)

# --- Step 1: Hash the input image ---
print(f"🔍 Hashing input image: {input_image}")
input_hash = nh.calculate_neuralhash(input_image)

# --- Step 2: Recursively compare with all subfolder images ---
found_similar = False
print(f"\n📂 Searching inside: {parent_folder}")

for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.lower().endswith(file_formats):
            image_path = os.path.join(root, file)
            try:
                folder_hash = nh.calculate_neuralhash(image_path)
                distance = hamming.calculate_hamming_distance_between_images(input_hash, folder_hash)

                if distance <= threshold:
                    print(f"\n✅ MATCH: {file} | Distance: {distance} | Folder: {os.path.relpath(root, parent_folder)}")
                    found_similar = True

                    # Show matched image
                    image = cv2.imread(image_path)
                    if image is not None:
                        cv2.imshow(f"Match: {file}", image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    print(f"❌ Not Similar: {file} | Distance: {distance}")
            except Exception as e:
                print(f"⚠️ Error processing {file}: {e}")

if not found_similar:
    print("\n🚫 No similar images found in any subfolder.")
