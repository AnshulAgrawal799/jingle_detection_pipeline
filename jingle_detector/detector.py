import numpy as np
import librosa
from scipy.signal import correlate
from librosa.sequence import dtw
import os
import subprocess
import warnings
from .audio_utils import load_audio, extract_chroma, extract_logmel
import matplotlib.pyplot as plt
from . import config


class DetectionResult:
    def __init__(self, filename, start_s, end_s, score, method):
        self.filename = filename
        self.start_s = start_s
        self.end_s = end_s
        self.score = score
        self.method = method

    def to_row(self):
        return [self.filename, self.start_s, self.end_s, self.score, self.method]


def slide_window(y, sr, win_len, hop_len):
    n_samples = len(y)
    win_samples = int(win_len * sr)
    hop_samples = int(hop_len * sr)
    for start in range(0, n_samples - win_samples + 1, hop_samples):
        end = start + win_samples
        yield start, end, y[start:end]


def detect_dtw(jingle_chroma, target_chroma, sr, win_len, hop_len, top_k=5, thresh=0.8, beta=1.0, debug=False):
    results = []
    raw_scores = []
    j_frames = jingle_chroma.shape[1]
    # Window step in frames
    step_frames = int(hop_len * sr)
    for start in range(0, target_chroma.shape[1] - j_frames + 1, step_frames):
        end = start + j_frames
        chunk_chroma = target_chroma[:, start:end]
        if chunk_chroma.shape[1] != j_frames:
            continue
        _, cost = dtw(jingle_chroma, chunk_chroma, subseq=True)
        # DTW cost normalization: divide by path length (frames)
        norm_cost = np.min(cost) / j_frames
        # Similarity mapping: sim = exp(-beta * norm_cost)
        sim = np.exp(-beta * norm_cost)
        raw_scores.append((start, end, norm_cost, sim))
        if sim >= thresh:
            results.append((start / sr, end / sr, sim))
    # Always return top_k highest similarity candidates
    top_candidates = sorted(raw_scores, key=lambda x: -x[3])[:top_k]
    top_results = [(s / sr, e / sr, sim) for s, e, _, sim in top_candidates]
    if debug:
        print("[DTW] Top 10 candidates (sim, start_s):", [(round(x[3], 3), round(
            x[0]/sr, 2)) for x in sorted(raw_scores, key=lambda x: -x[3])[:10]])
        max_sim = max(raw_scores, key=lambda x: x[3])
        print(
            f"[DTW] Max similarity: {max_sim[3]:.3f} at {max_sim[0]/sr:.2f}s")
    return top_results, raw_scores


def detect_corr(jingle_logmel, target_logmel, sr, win_len, hop_len, top_k=5, thresh=0.8, debug=False):
    results = []
    raw_scores = []
    j_len = jingle_logmel.shape[1]
    step_frames = int(hop_len * sr)
    for start in range(0, target_logmel.shape[1] - j_len + 1, step_frames):
        end = start + j_len
        chunk_logmel = target_logmel[:, start:end]
        if chunk_logmel.shape[1] != j_len:
            continue
        # Normalized cross-correlation: divide by norms, mode='valid'
        norm_corr = []
        for i in range(jingle_logmel.shape[0]):
            a = jingle_logmel[i]
            b = chunk_logmel[i]
            # Compute normalized correlation
            if np.std(a) == 0 or np.std(b) == 0:
                r = 0.0
            else:
                r = np.correlate((a - np.mean(a)) / np.std(a),
                                 (b - np.mean(b)) / np.std(b), mode='valid')[0] / len(a)
            norm_corr.append(r)
        score = np.mean(norm_corr)
        # Map [-1,1] to [0,1]
        sim = (score + 1) / 2
        raw_scores.append((start, end, score, sim))
        if sim >= thresh:
            results.append((start / sr, end / sr, sim))
    # Always return top_k highest similarity candidates
    top_candidates = sorted(raw_scores, key=lambda x: -x[3])[:top_k]
    top_results = [(s / sr, e / sr, sim) for s, e, _, sim in top_candidates]
    if debug:
        print("[Corr] Top 10 candidates (sim, start_s):", [(round(x[3], 3), round(
            x[0]/sr, 2)) for x in sorted(raw_scores, key=lambda x: -x[3])[:10]])
        max_sim = max(raw_scores, key=lambda x: x[3])
        print(
            f"[Corr] Max similarity: {max_sim[3]:.3f} at {max_sim[0]/sr:.2f}s")
    return top_results, raw_scores
