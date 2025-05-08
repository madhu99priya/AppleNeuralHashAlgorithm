import os
import sys
import cv2
from preprocessing_mediapipe import NeuralHash, Hamming

# --- Configurable values ---
folder_path = "./images"   
file_formats = (".jpg", ".png")
threshold = 15
split_char = "_"

# --- Command-line input check ---
if len(sys.argv) != 2:
    print("❌ Usage: python compare_image_to_folder.py <input_image_path>")
    sys.exit(1)

input_image = sys.argv[1]

if not os.path.isfile(input_image):
    print(f"❌ Error: File not found - {input_image}")
    sys.exit(1)

# --- Initialize classes ---
nh = NeuralHash()
hamming = Hamming(split_char=split_char, output_format=1)

# --- Step 1: Hash input image ---
print(f"🔍 Hashing input image: {input_image}")
input_name = os.path.basename(input_image)
input_hash = nh.calculate_neuralhash(input_image)

# --- Step 2: Compare with folder ---
print(f"\n📂 Comparing against images in: {folder_path}")
found_similar = False

for file in os.listdir(folder_path):
    if file.lower().endswith(file_formats):
        path = os.path.join(folder_path, file)
        try:
            folder_hash = nh.calculate_neuralhash(path)
            distance = hamming.calculate_hamming_distance_between_images(input_hash, folder_hash)
            if distance <= threshold:
                print(f"✅ MATCH: {file} | Distance: {distance}")
                found_similar = True

                # Display the matched image using OpenCV
                image = cv2.imread(path)
                if image is not None:
                    cv2.imshow(f"Match: {file} | Distance: {distance}", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print(f"❌ Not Similar: {file} | Distance: {distance}")
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

if not found_similar:
    print("\n🚫 No similar images found under threshold.")
