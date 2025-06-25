import numpy as np

def analyze_neuralhash_hyperplanes(file_path: str):
    """Reads and prints all 96 hyperplanes from NeuralHash .dat file (each of 128 dimensions)."""
    
    # Load binary float32 data
    data = np.fromfile(file_path, dtype=np.float32)
    expected_size = 128 * 96

    # Check if the data is the correct length
    if len(data) < expected_size:
        raise ValueError(f"Data too short. Expected {expected_size}, but got {len(data)}")
    elif len(data) > expected_size:
        print(f"Warning: Extra data detected ({len(data)} values). Trimming to {expected_size}.")
        data = data[:expected_size]

    # Reshape into 128x96 matrix (columns are hyperplanes)
    M = data.reshape(128, 96)

    print(f"Loaded {M.shape[1]} hyperplanes of dimension {M.shape[0]}.\n")

    for i in range(96):
        hyperplane = M[:, i]
        print(f"Hyperplane {i+1} (128D):\n{hyperplane}\n")

    return M

# Run when executed directly
if __name__ == "__main__":
    analyze_neuralhash_hyperplanes("model/neuralhash_128x96_seed1.dat")
