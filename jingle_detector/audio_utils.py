import librosa
import numpy as np


def load_audio(path, sr=22050):
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    y = y / np.max(np.abs(y))
    return y, sr


def extract_chroma(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return chroma


def extract_logmel(y, sr, n_mels=64):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S
