import os
import time
from preprocessing_mediapipe import NeuralHash, Hamming 

# -------- CONFIGURATION -------- #
image_dir = "./images"  
file_formats = (".jpg", ".png")    # File formats allowed
threshold = 15                   
split_char = "_"  
# -------------------------------- #


start_time = time.time()

# Get all image paths
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
               if f.lower().endswith(file_formats)]

# Initialize NeuralHash and Hamming tools
nh = NeuralHash()
hamming_calc = Hamming(split_char=split_char, threshold=threshold, output_format=1)  # binary format

# Step 1: Calculate NeuralHash for all images
print("📷 Generating hashes...")
for image_path in image_paths:
    image_name = os.path.basename(image_path)
    try:
        neural_hash = nh.calculate_neuralhash(image_path)
        hamming_calc.insert_neuralhash(image_name, neural_hash)
    except Exception as e:
        print(f"❌ Failed to process {image_name}: {e}")

# Step 2: Calculate pairwise Hamming distances
print("\n📏 Calculating Hamming distances...")
hamming_calc.calculate_hamming_distances()

# Step 3: Group images by similarity
print("\n🔗 Grouping similar images...")
visited = set()
clusters = []

for img1 in hamming_calc.hamming_distances:
    if img1 in visited:
        continue

    cluster = {img1}
    queue = [img1]

    while queue:
        current = queue.pop()
        visited.add(current)

        for img2, (_, _, dist_bin) in hamming_calc.hamming_distances[current].items():
            if dist_bin <= threshold and img2 not in visited:
                cluster.add(img2)
                queue.append(img2)

    if len(cluster) > 1:  # Ignore singletons
        clusters.append(sorted(cluster))

# Step 4: Print results
if not clusters:
    print("\n🚫 No similar images found under the current threshold.")
else:
    print(f"\n📦 Found {len(clusters)} groups of similar images (Hamming distance ≤ {threshold}):\n")
    for i, group in enumerate(clusters, 1):
        print(f"Group {i}:")
        for img in group:
            print(f"  - {img}")
        print()


end_time = time.time()

total_time = end_time-start_time

print(f"\n⏰ Total time taken: {total_time:.2f} seconds")