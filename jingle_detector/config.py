
"""
Configuration constants for jingle detection pipeline.
HOP_LENGTH: Feature extraction hop length (samples).
DEFAULT_BETA: Controls how quickly similarity decays with DTW avg distance.
beta controls how quickly similarity decays with DTW avg distance.
"""

HOP_LENGTH = 512
DEFAULT_BETA = 1.0

# Typical audio processing configuration variables
DEFAULT_SR = 22050  # Standard sample rate for audio analysis
DTW_THRESH = 0.3    # Default threshold for DTW similarity
CORR_THRESH = 0.3   # Default threshold for correlation similarity
WIN_LEN = 2048      # Window length for feature extraction
N_MELS = 128        # Number of Mel bands for spectrogram
