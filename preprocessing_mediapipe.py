import os
import sys
import math
import operator
import itertools
import collections
import json
import csv
import Levenshtein
import numpy as np
from PIL import Image
from onnxruntime import InferenceSession
from xml.dom import minidom
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

# Enable colored output for Windows
if sys.platform in ["cygwin", "win32"]:
    import colorama
    colorama.init()

# ANSI text formatting
END = "\x1b[0m"
BOLD = "\x1b[1m"
RED = "\x1b[31m"
BRIGHT_RED_BG = "\x1b[101m"
GREEN = "\x1b[32m"
BLUE = "\x1b[34m"

# Sorting alternatives for Experiment 2
DIST_SAME = 0
DIST_OTHERS = 1
SD_SAME = 2
SD_OTHERS = 3

# Constants for getting closest/furthest Hamming Distance
CLOSEST = 0
FURTHEST = 1


class NeuralHash:
    def __init__(self, model_path=None):
        # Load ONNX model
        if model_path is None:
            self.model_path = f"{os.getcwd()}/retrain/newModel.onnx"
        else:
            self.model_path = model_path
            
        print(f"Loading model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        self.session = InferenceSession(self.model_path)
        
        # Get input details to determine expected shape
        input_details = self.session.get_inputs()[0]
        self.input_shape = input_details.shape
        
        # Extract expected dimensions (assuming NCHW format)
        if len(self.input_shape) == 4:
            _, _, self.expected_height, self.expected_width = self.input_shape
        else:
            # Default to 384x384 if can't determine from model
            self.expected_height, self.expected_width = 384, 384
            
        print(f"Model expects input dimensions: {self.expected_height}x{self.expected_width}")

        # Load output hash matrix
        try:
            # Get directory of the model file
            model_dir = os.path.dirname(self.model_path)
            seed1_path = os.path.join(model_dir, "neuralhash_128x96_seed1.dat")
            
            if not os.path.exists(seed1_path):
                raise FileNotFoundError(f"Seed file not found at {seed1_path}")
                
            self.seed1 = open(seed1_path, 'rb').read()[128:]
            self.seed1 = np.frombuffer(self.seed1, dtype=np.float32)
            self.seed1 = self.seed1.reshape([96, 128])
        except Exception as e:
            print(f"Error loading seed file: {e}")
            sys.exit(1)

    def calculate_neuralhash(self, image_path):
        """Calculate neuralhash of the image at image_path"""
        try:
            print(f"Processing image: {image_path}")
            arr = self.im2array(image_path)
            
            # Run model
            inputs = {self.session.get_inputs()[0].name: arr}
            outs = self.session.run(None, inputs)

            # Convert model output to hex hash
            hash_output = self.seed1.dot(outs[0].flatten())
            hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
            hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)

            return hash_hex, hash_bits
        except Exception as e:
            print(f"{RED}Error calculating neural hash: {e}{END}")
            return None

    ## using Mediapipe Face Detection
    def im2array(self, image_path):
        """Preprocess only the detected face region using Mediapipe"""
        try:
            image = Image.open(image_path).convert('RGB')
            img_np = np.array(image)

            mp_face_detection = mp.solutions.face_detection
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

                if not results.detections:
                    print(f"No face detected in {image_path}. Proceeding with the whole image.")
                    face_crop = img_np
                else:
                    # Take first detection
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box

                    ih, iw, _ = img_np.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)

                    # Expand the bounding box slightly
                    expansion = 0.2
                    x_exp = int(w * expansion / 2)
                    y_exp = int(h * expansion / 2)

                    x_new = max(0, x - x_exp)
                    y_new = max(0, y - y_exp)
                    w_new = min(iw - x_new, w + 2 * x_exp)
                    h_new = min(ih - y_new, h + 2 * y_exp)

                    face_crop = img_np[y_new:y_new+h_new, x_new:x_new+w_new]

            # Resize cropped face to expected dimensions
            face_crop = cv2.resize(face_crop, (self.expected_width, self.expected_height))

            # Normalize to [-1, 1]
            arr = face_crop.astype(np.float32) / 255.0
            arr = arr * 2.0 - 1.0

            return arr.transpose(2, 0, 1).reshape([1, 3, self.expected_height, self.expected_width])
        except Exception as e:
            print(f"{RED}Error in image preprocessing: {e}{END}")
            raise


class Hamming:
    def __init__(self, split_char, threshold=0, output_format=0, save_dict=False, load_dict=False):
        self.split_char = split_char  # Delimiter for splitting image file names
        self.threshold = threshold  # Threshold hamming distance for considering two images as the same subject
        self.hash_dict = {}  # Dictionary of image hashes. {img: (hash_hex, hash_bin)}
        self.hamming_distances = {}  # Dictionary of hamming distances between image hashes
        self.rates = {}  # Dictionary of subjects' accept and reject rates
        self.output_format = output_format  # Configure output of NeuralHashes to hex or binary. 0 = hex, 1 = bin
        self.save_dict_json = save_dict  # Toggle saving of hash_dict and hamming_distances dicts to json
        self.load_dict_json = load_dict  # Toggle loading of hash_dict and hamming_distances dicts from json
        self.spacing = None  # For aligning output to a grid
        self.hl = None  # Horizontal line
        self.hl_short = None  # Short horizontal line

    def insert_neuralhash(self, image, nhash):
        """Append neuralhash to self.hash_dict"""
        self.hash_dict[image] = nhash

    def set_printing_params(self):
        """
        Set parameters for aligning output
        according to file names of images
        """
        if not self.hash_dict:
            return
            
        self.spacing = max([len(i) for i in self.hash_dict]) + 1
        self.hl = f"\n{'-' * (self.spacing + 28 + (self.output_format * 72))}\n"
        self.hl_short = f"\n{'-' * (self.spacing + 28)}\n"

    def save_dict(self, hash_dict="hashes.json", hamming_dict="hamming.json"):
        """Save self.hash_dict and self.hamming_distances to json files"""
        with open(hash_dict, 'w') as handle:
            json.dump(self.hash_dict, handle)
        print(f"Saved hash_dict as {hash_dict}")

        with open(hamming_dict, 'w') as handle:
            json.dump(self.hamming_distances, handle)
        print(f"Saved hamming_distances as {hamming_dict}")

    def load_dict(self, hash_dict="hashes.json", hamming_dict="hamming.json"):
        """Load self.hash_dict and self.hamming_distances from json files"""
        files = os.listdir(os.getcwd())

        for f in files:
            if "hashes" in f and f.endswith(".json"):
                hash_dict = f
            elif "hamming" in f and f.endswith(".json"):
                hamming_dict = f

        with open(hash_dict, 'r') as handle:
            self.hash_dict = json.load(handle)
        print(f"Loaded hash_dict from {hash_dict}")

        with open(hamming_dict, 'r') as handle:
            self.hamming_distances = json.load(handle)
        print(f"Loaded hamming_distances from {hamming_dict}")

    def calculate_hamming_distance_between_images(self, hash1, hash2):
        """Calculate and return the Hamming distance between two image hashes"""
        if hash1 is None or hash2 is None:
            print(f"{RED}Error: Cannot calculate Hamming distance - one or both hashes are None{END}")
            return None
            
        try:
            # hamming_dist_hex = Levenshtein.hamming(hash1[0], hash2[0])
            hamming_dist_bin = Levenshtein.hamming(hash1[1], hash2[1])
            print(f"\n🔹 Hamming Distance (Binary): {hamming_dist_bin}\n")
            return hamming_dist_bin
        except Exception as e:
            print(f"{RED}Error calculating Hamming distance: {e}{END}")
            return None


def main():
    # Configure file format and delimiter
    file_format = (".ppm", ".jpg", ".png", ".jpeg")
    split_char = '_'  # Delimiter for splitting image file names

    # Parse command line arguments
    if len(sys.argv) not in [2, 3]:
        print(f"{BLUE}Usage:{END}")
        print("  python preprocessing_mediapipe.py <image1_path>           # Prints Neural Hash for one image")
        print("  python preprocessing_mediapipe.py <image1_path> <image2_path>  # Prints Neural Hashes and Hamming Distance")
        return

    # Search for model files in current directory and parent directories
    model_path = None
    current_dir = os.getcwd()
    
    # Define possible model filenames
    model_filenames = [ "newModel.onnx", "converted_neuralhash.onnx", "model.onnx"]
    
    # Check in current directory and its model subfolder
    for filename in model_filenames:
        # Check in /model subdirectory
        if os.path.exists(os.path.join(current_dir, "model", filename)):
            model_path = os.path.join(current_dir, "model", filename)
            break
        # Check in current directory
        elif os.path.exists(os.path.join(current_dir, filename)):
            model_path = os.path.join(current_dir, filename)
            break
        # Check one level up
        elif os.path.exists(os.path.join(current_dir, "..", "model", filename)):
            model_path = os.path.join(current_dir, "..", "model", filename)
            break
    
    # If no model found, use default path (will likely fail but with proper error)
    if model_path is None:
        print(f"{RED}Warning: Could not find model file. Will try default path.{END}")
    
    # Initialize classes with proper error handling
    try:
        NHash = NeuralHash(model_path)
        hamming = Hamming(split_char, output_format=1)  # Using binary format (1) for better accuracy
    except Exception as e:
        print(f"{RED}Failed to initialize: {e}{END}")
        return

    # Process the first image
    image1_path = sys.argv[1]
    if not os.path.isfile(image1_path):
        print(f"{RED}Error: File '{image1_path}' does not exist.{END}")
        return

    def process_image(image_path):
        file_name = os.path.basename(image_path)
        try:
            print(f"\n{BLUE}Processing file: {file_name}{END}")
            neural_hash = NHash.calculate_neuralhash(image_path)
            if neural_hash:
                print(f"{GREEN}🟢 Neural Hash for {file_name}:{END}")
                print(f"   - Hex: {neural_hash[0]}")
                print(f"   - Binary: {neural_hash[1][:64]}... (total length: {len(neural_hash[1])})")
                return neural_hash
            else:
                print(f"{RED}Failed to generate Neural Hash for {file_name}{END}")
                return None
        except Exception as e:
            print(f"{RED}Error processing {file_name}: {e}{END}")
            return None

    # Process first image
    hash1 = process_image(image1_path)

    # If only one image provided, exit
    if len(sys.argv) == 2:
        if hash1:
            print(f"\n{GREEN}✅ Process completed successfully.{END}")
        else:
            print(f"\n{RED}❌ Process failed.{END}")
        return

    # Process second image and calculate Hamming distance
    image2_path = sys.argv[2]
    if not os.path.isfile(image2_path):
        print(f"{RED}Error: File '{image2_path}' does not exist.{END}")
        return

    hash2 = process_image(image2_path)
    
    # Calculate Hamming distance if both hashes were generated successfully
    if hash1 and hash2:
        hamming_distance = hamming.calculate_hamming_distance_between_images(hash1, hash2)
        if hamming_distance is not None:
            print(f"{BLUE}🔹 Hamming Distance between {os.path.basename(image1_path)} and {os.path.basename(image2_path)}:{END}")
            print(f"   - Binary Distance: {hamming_distance}")
            
            # Provide a simple threshold-based similarity assessment
            max_distance = len(hash1[1])  # Length of binary hash
            similarity_percent = ((max_distance - hamming_distance) / max_distance) * 100
            
            print(f"\n{BLUE}Similarity Assessment:{END}")
            print(f"   - Similarity: {similarity_percent:.2f}%")
            
            if similarity_percent > 95:
                print(f"   - Interpretation: {GREEN}Very likely the same person{END}")
            elif similarity_percent > 90:
                print(f"   - Interpretation: {GREEN}Likely the same person{END}")
            elif similarity_percent > 80:
                print(f"   - Interpretation: {BLUE}Possibly the same person{END}")
            else:
                print(f"   - Interpretation: {RED}Likely different people{END}")
        
        print(f"\n{GREEN}✅ Process completed.{END}")
    else:
        print(f"\n{RED}❌ Unable to calculate Hamming distance due to hash generation failure.{END}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{RED}Unhandled exception: {e}{END}")
        import traceback
        traceback.print_exc()