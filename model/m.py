import numpy as np

def analyze_neuralhash_dat(file_path: str):
    """Analyzes NeuralHash .dat file for research purposes"""
    # Read binary data
    data = np.fromfile('model\neuralhash_128x96_seed1.dat', dtype=np.float32)
    
    # Reshape to 128x96 matrix
    M = data.reshape(128, 96)
    
    # Compute statistics
    stats = {
        "shape": M.shape,
        "mean": np.mean(M),
        "std": np.std(M),
        "min": np.min(M),
        "max": np.max(M),
        "sample_values": M[:5, :5].tolist()  # Top-left 5x5 submatrix
    }
    return stats

# Usage (replace with actual file path)
# stats = analyze_neuralhash_dat("neuralhash_128x96_seed1.dat")
