import subprocess
import tempfile
import numpy as np
import librosa
from . import config


def extract_chroma(y, sr, hop_length=None, n_fft=None):
    """
    Compute chroma features.

    Parameters
    ----------
    y : np.ndarray
        Audio time series (mono).
    sr : int
        Sample rate.
    hop_length : int or None
        Hop length in samples. If None, use config.HOP_LENGTH.
    n_fft : int or None
        FFT window size. If None, use 4 * hop_length as a reasonable default.

    Returns
    -------
    chroma : np.ndarray, shape=(n_chroma, n_frames)
        Chroma feature matrix.
    """
    if hop_length is None:
        hop_length = getattr(config, "HOP_LENGTH", 512)
    if n_fft is None:
        n_fft = hop_length * 4

    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    return chroma


def extract_logmel(y, sr, hop_length=None, n_mels=128, n_fft=None):
    """
    Compute log-mel spectrogram (in dB).

    Parameters
    ----------
    y : np.ndarray
        Audio time series (mono).
    sr : int
        Sample rate.
    hop_length : int or None
        Hop length in samples. If None, use config.HOP_LENGTH.
    n_mels : int
        Number of Mel bands.
    n_fft : int or None
        FFT window size. If None, use 4 * hop_length as a reasonable default.

    Returns
    -------
    log_mel : np.ndarray, shape=(n_mels, n_frames)
        Log-scaled Mel spectrogram in dB.
    """
    if hop_length is None:
        hop_length = getattr(config, "HOP_LENGTH", 512)
    if n_fft is None:
        n_fft = hop_length * 4

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel = librosa.power_to_db(S, ref=np.max)
    return log_mel


def load_audio(path, sr=None, mono=True, reencode_bad_mp3=False):
    """
    Load audio, optionally re-encoding MP3 to WAV for robust reading.
    Returns (y, sr, used_path).
    """
    if reencode_bad_mp3 and path.lower().endswith('.mp3'):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp_path = tmp.name
        tmp.close()
        ffmpeg_sr = sr if sr is not None else 22050
        print(f"Re-encoding MP3 to WAV at {tmp_path}")
        cmd = [
            'ffmpeg', '-y', '-i', path,
            '-ar', str(ffmpeg_sr), '-ac', '1', '-c:a', 'pcm_s16le', tmp_path
        ]
        subprocess.run(cmd, check=True)
        y, sr = librosa.load(tmp_path, sr=sr, mono=mono)
        used_path = tmp_path
    else:
        y, sr = librosa.load(path, sr=sr, mono=mono)
        used_path = path
    return y, sr, used_path


def seconds_to_samples(s, sr):
    return int(round(s * sr))


def samples_to_seconds(n, sr):
    return float(n) / sr


def extract_logmel(y, sr, n_mels=64):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S
