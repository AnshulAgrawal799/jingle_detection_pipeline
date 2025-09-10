from . import audio_utils, config
import os
import scipy.spatial.distance
import scipy.signal
import soundfile as sf
import librosa
import numpy as np
from dataclasses import dataclass

# --- Compatibility wrappers for legacy __main__.py imports ---


@dataclass
class DetectionResult:
    method: str
    start_s: float
    score: float
    rank: int = 0


def detect_dtw(jingle_path, target_path, top_k=5, threshold_dtw=0.3, reencode_bad_mp3=False):
    """
    Compatibility wrapper for DTW detection. Returns list of DetectionResult.
    """
    # Use process_target to get candidates
    results = []

    def collect_dtw(jingle_path, target_paths, **kwargs):
        # Only collect DTW results for the first target
        template_audio, sr, _ = audio_utils.load_audio(
            jingle_path, reencode_bad_mp3=reencode_bad_mp3)
        hop_length = config.HOP_LENGTH
        # Use both chroma and mel features for robustness
        chroma = librosa.feature.chroma_stft(
            y=template_audio, sr=sr, hop_length=hop_length, n_fft=hop_length*2)
        mel = librosa.feature.melspectrogram(
            y=template_audio, sr=sr, n_fft=hop_length*2, hop_length=hop_length, n_mels=config.N_MELS)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)
        target_audio, sr_t, _ = audio_utils.load_audio(
            target_paths[0], sr=sr, reencode_bad_mp3=reencode_bad_mp3)
        target_chroma = librosa.feature.chroma_stft(
            y=target_audio, sr=sr, hop_length=hop_length, n_fft=hop_length*2)
        target_mel = librosa.feature.melspectrogram(
            y=target_audio, sr=sr, n_fft=hop_length*2, hop_length=hop_length, n_mels=config.N_MELS)
        target_mel = librosa.power_to_db(target_mel, ref=np.max)
        target_mel = (target_mel - np.mean(target_mel)) / \
            (np.std(target_mel) + 1e-6)
        template_frames = chroma.shape[1]
        n_target_frames = target_chroma.shape[1]
        # Diagnostic output
        print(
            f"[DTW] template chroma shape: {chroma.shape}, mel shape: {mel.shape}")
        print(
            f"[DTW] target chroma shape: {target_chroma.shape}, mel shape: {target_mel.shape}")
        window_frames = template_frames
        step_frames = max(1, int(round(0.1 * sr / hop_length)))
        dtw_candidates = []
        all_scores = []
        for f in np.arange(0, n_target_frames - window_frames + 1, step_frames):
            f = int(f)
            W_chroma = target_chroma[:, f:f+template_frames]
            W_mel = target_mel[:, f:f+template_frames]
            if W_chroma.shape[1] != template_frames or W_mel.shape[1] != mel.shape[1]:
                continue
            # Combine chroma and mel features
            combined_template = np.vstack([chroma, mel])
            combined_window = np.vstack([W_chroma, W_mel])
            D_local = scipy.spatial.distance.cdist(
                combined_template.T, combined_window.T, metric='euclidean')
            D_acc, wp = librosa.sequence.dtw(C=D_local)
            wp = np.array(wp)
            local_costs = D_local[wp[:, 0], wp[:, 1]]
            raw_cost = local_costs.sum()
            path_len = len(wp)
            norm_distance = raw_cost / float(path_len)
            beta = getattr(config, 'DEFAULT_BETA', 1.0)
            sim = np.exp(-beta * norm_distance)
            all_scores.append(sim)
            if sim >= threshold_dtw:
                dtw_candidates.append(
                    {'start_s': (f * hop_length) / sr, 'sim': sim})
        print(f"[DTW] All similarity scores: {all_scores}")
        print(f"[DTW] Candidates above threshold: {dtw_candidates}")
        dtw_sorted = sorted(dtw_candidates, key=lambda x: -x['sim'])[:top_k]
        for i, c in enumerate(dtw_sorted):
            results.append(DetectionResult(
                method='dtw', start_s=c['start_s'], score=c['sim'], rank=i+1))
    collect_dtw(jingle_path, [target_path])
    return results


def detect_corr(jingle_path, target_path, top_k=5, threshold_corr=0.3, reencode_bad_mp3=False):
    """
    Compatibility wrapper for correlation detection. Returns list of DetectionResult.
    """
    results = []

    def collect_corr(jingle_path, target_paths, **kwargs):
        template_audio, sr, _ = audio_utils.load_audio(
            jingle_path, reencode_bad_mp3=reencode_bad_mp3)
        target_audio, sr_t, _ = audio_utils.load_audio(
            target_paths[0], sr=sr, reencode_bad_mp3=reencode_bad_mp3)
        step_samples = max(1, int(round(0.1 * sr)))
        template_samples = len(template_audio)
        corr_candidates = []
        all_corr_scores = []
        for s in range(0, len(target_audio) - template_samples + 1, step_samples):
            window = target_audio[s:s+template_samples]
            if len(window) != template_samples:
                continue
            r = np.correlate(window, template_audio, mode='valid')
            t_norm = np.linalg.norm(template_audio)
            window_sq_sum = scipy.signal.convolve(
                window * window, np.ones(len(template_audio)), mode='valid')
            x_norms = np.sqrt(window_sq_sum)
            denom = t_norm * x_norms
            denom[denom == 0] = np.finfo(float).eps
            r_norm = r / denom
            r_mapped = (r_norm + 1.0) / 2.0
            all_corr_scores.append(float(r_mapped))
            if r_mapped >= threshold_corr:
                corr_candidates.append(
                    {'start_s': s / sr, 'r_mapped': float(r_mapped)})
        print(f"[CORR] All correlation scores: {all_corr_scores}")
        print(f"[CORR] Candidates above threshold: {corr_candidates}")
        corr_sorted = sorted(
            corr_candidates, key=lambda x: -x['r_mapped'])[:top_k]
        for i, c in enumerate(corr_sorted):
            results.append(DetectionResult(
                method='corr', start_s=c['start_s'], score=c['r_mapped'], rank=i+1))
    collect_corr(jingle_path, [target_path])
    return results


# --- Smoke test for wrappers ---
if __name__ == "__main__":
    # Generate synthetic signals
    sr = 22050
    t = np.linspace(0, 1, sr)
    jingle = np.sin(2 * np.pi * 440 * t)
    target = np.concatenate([np.zeros(sr//2), jingle, np.zeros(sr//2)])
    import tempfile
    jingle_path = tempfile.mktemp(suffix="_jingle.wav")
    target_path = tempfile.mktemp(suffix="_target.wav")
    sf.write(jingle_path, jingle, sr)
    sf.write(target_path, target, sr)
    print("DTW results:", detect_dtw(jingle_path, target_path))
    print("Corr results:", detect_corr(jingle_path, target_path))


def process_target(jingle_path, target_paths, plot_dir, output_csv,
                   debug=False, top_k=5, reencode_bad_mp3=False,
                   threshold_dtw=0.3, threshold_corr=0.3,
                   window_step_s=0.1, window_size_s=None):
    print(
        f"[DEBUG] Entered process_target with jingle_path={jingle_path}, target_paths={target_paths}, output_csv={output_csv}, debug={debug}")
    # Load jingle template
    template_audio, sr, template_path = audio_utils.load_audio(
        jingle_path, reencode_bad_mp3=reencode_bad_mp3)
    template_length_s = len(template_audio) / sr
    hop_length = config.HOP_LENGTH
    chroma = librosa.feature.chroma_stft(
        y=template_audio, sr=sr, hop_length=hop_length, n_fft=hop_length*2)
    mel = librosa.feature.melspectrogram(
        y=template_audio, sr=sr, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel)
    template_frames = chroma.shape[1]

    all_results = []
    for target_path in target_paths:
        print(f"[DEBUG] Processing target_path: {target_path}")
        target_audio, sr_t, used_path = audio_utils.load_audio(
            target_path, sr=sr, reencode_bad_mp3=reencode_bad_mp3)
        print("[DEBUG] Finished loading target audio")
        assert sr_t == sr, f"Sample rate mismatch: {sr_t} != {sr}"
        target_length_s = len(target_audio) / sr
        target_chroma = librosa.feature.chroma_stft(
            y=target_audio, sr=sr, hop_length=hop_length, n_fft=hop_length*2)
        target_mel = librosa.feature.melspectrogram(
            y=target_audio, sr=sr, hop_length=hop_length)
        target_log_mel = librosa.power_to_db(target_mel)
        print("[DEBUG] Finished extracting chroma and mel features")
        n_target_frames = target_chroma.shape[1]

        # Sliding window setup
        step_frames = max(1, int(round(window_step_s * sr / hop_length)))
        if window_size_s is None:
            window_size_s = template_length_s + 0.05
        window_frames = max(1, int(round(window_size_s * sr / hop_length)))
        assert step_frames >= 1 and window_frames >= 1, "step_frames and window_frames must be >= 1"

        print("[DEBUG] Starting DTW sliding window")
        # DTW sliding window
        dtw_candidates = []
        total_windows = n_target_frames - window_frames + 1
        for idx, f in enumerate(np.arange(0, total_windows, step_frames)):
            f = int(f)
            if idx % 100 == 0:
                print(f"[DEBUG] DTW window {idx}/{total_windows}")
            W_chroma = target_chroma[:, f:f+template_frames]
            if W_chroma.shape[1] != template_frames:
                continue
            print(
                f"[DEBUG] DTW input shapes: chroma.T={chroma.T.shape}, W_chroma.T={W_chroma.T.shape}")
            D_local = scipy.spatial.distance.cdist(
                chroma.T, W_chroma.T, metric='euclidean')
            D_acc, wp = librosa.sequence.dtw(C=D_local)
            wp = np.array(wp)
            local_costs = D_local[wp[:, 0], wp[:, 1]]
            raw_cost = local_costs.sum()
            path_len = len(wp)
            norm_distance = raw_cost / float(path_len)
            beta = config.DEFAULT_BETA
            sim = np.exp(-beta * norm_distance)
            dtw_candidates.append({
                'start_frame': f,
                'start_s': (f * hop_length) / sr,
                'raw_cost': raw_cost,
                'path_len': path_len,
                'norm_distance': norm_distance,
                'sim': sim
            })
            # Remove debug break to process all windows
            # if idx > 10:
            #     print("[DEBUG] Breaking after 10 DTW windows for diagnosis.")
            #     break
        print("[DEBUG] Finished DTW sliding window")

        print("[DEBUG] Starting cross-correlation sliding window")
        # Cross-correlation sliding window
        step_samples = max(1, int(round(window_step_s * sr)))
        template_samples = len(template_audio)
        corr_candidates = []
        for idx, s in enumerate(range(0, len(target_audio) - template_samples + 1, step_samples)):
            if idx % 1 == 0:
                print(f"[DEBUG] CORR window {idx}")
            window = target_audio[s:s+template_samples]
            if len(window) != template_samples:
                continue
            r = np.correlate(window, template_audio, mode='valid')
            t_norm = np.linalg.norm(template_audio)
            window_sq_sum = scipy.signal.convolve(
                window * window, np.ones(len(template_audio)), mode='valid')
            x_norms = np.sqrt(window_sq_sum)
            denom = t_norm * x_norms
            denom[denom == 0] = np.finfo(float).eps
            r_norm = r / denom
            r_mapped = (r_norm + 1.0) / 2.0
            corr_candidates.append({
                'start_sample': s,
                'start_s': s / sr,
                'r': float(r[0]),
                'r_norm': float(r_norm[0]),
                'r_mapped': float(r_mapped[0])
            })
            if debug and idx > 10:
                print("[DEBUG] Breaking after 10 CORR windows for diagnosis.")
                break
        print("[DEBUG] Finished cross-correlation sliding window")

        print("[DEBUG] Sorting and selecting top-K candidates")
        # Sort and select top-K
        dtw_sorted = sorted(dtw_candidates, key=lambda x: -x['sim'])[:top_k]
        corr_sorted = sorted(
            corr_candidates, key=lambda x: -x['r_mapped'])[:top_k]

        # Debug logging
        print(f"File: {target_path}")
        print(f"Sample rate: {sr}, length: {target_length_s:.2f}s")
        print(f"Template length: {template_length_s:.2f}s")
        print(
            f"chroma.shape: {target_chroma.shape}, log_mel.shape: {target_log_mel.shape}")
        print("Top-10 DTW candidates:")
        for i, c in enumerate(dtw_sorted):
            print(
                f"  {i+1:2d}: raw_cost={c['raw_cost']:.2f}, path_len={c['path_len']}, norm_dist={c['norm_distance']:.3f}, sim={c['sim']:.3f}, start_s={c['start_s']:.2f}")

        print("Top-10 correlation candidates:")
        for i, c in enumerate(corr_sorted):
            print(
                f"  {i+1:2d}: r={c['r']:.3f}, r_norm={c['r_norm']:.3f}, r_mapped={c['r_mapped']:.3f}, start_s={c['start_s']:.2f}")

        # Global maxima
        if dtw_sorted:
            best_dtw = dtw_sorted[0]
            print(
                f"Best DTW sim={best_dtw['sim']:.2f} at {best_dtw['start_s']:.2f}s")
        if corr_sorted:
            best_corr = corr_sorted[0]
            print(
                f"Best corr={best_corr['r_mapped']:.2f} at {best_corr['start_s']:.2f}s")

        # Write snippets
        snippets_dir = os.path.join('output', 'snippets')
        os.makedirs(snippets_dir, exist_ok=True)
        base = os.path.basename(target_path)
        snippet_duration = template_length_s + 0.02
        snippet_samples = audio_utils.seconds_to_samples(snippet_duration, sr)
        for i, c in enumerate(dtw_sorted):
            start_sample = audio_utils.seconds_to_samples(c['start_s'], sr)
            end_sample = start_sample + snippet_samples
            snippet = target_audio[start_sample:end_sample]
            outpath = os.path.join(
                snippets_dir, f"{base}__cand_{i+1:02d}_dtw.wav")
            sf.write(outpath, snippet, sr)
        for i, c in enumerate(corr_sorted):
            start_sample = c['start_sample']
            end_sample = start_sample + snippet_samples
            snippet = target_audio[start_sample:end_sample]
            outpath = os.path.join(
                snippets_dir, f"{base}__cand_{i+1:02d}_corr.wav")
            sf.write(outpath, snippet, sr)

        # All candidates at 0s warning
        if all(c['start_s'] == 0.0 for c in dtw_sorted) and all(c['start_s'] == 0.0 for c in corr_sorted):
            print("WARNING: All candidates at 0s — check sliding window step_frames/window_frames and hop_length consistency.")

        # Accumulate results for CSV
        # Always write at least one row per file (even if no candidates)
        if dtw_sorted or corr_sorted:
            for i, c in enumerate(dtw_sorted):
                all_results.append([
                    os.path.basename(target_path), 'dtw', i+1, c['start_s'], c['sim']])
            for i, c in enumerate(corr_sorted):
                all_results.append([
                    os.path.basename(target_path), 'corr', i+1, c['start_s'], c['r_mapped']])
        else:
            all_results.append([
                os.path.basename(target_path), 'none', '', '', 0.0])
            print(
                f"[DEBUG] No candidates found for {os.path.basename(target_path)}; will write 'none' row to CSV.")

    # Write CSV once after all targets
    import csv
    debug_csv = output_csv if debug else output_csv.replace(
        '.csv', '_debug.csv')
    output_dir = os.path.dirname(debug_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"[DEBUG] About to write debug CSV to: {debug_csv}")
    with open(debug_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'method', 'rank', 'start_s', 'score'])
        for row in all_results:
            writer.writerow(row)
    print(f"[DEBUG] Wrote debug CSV to: {debug_csv}")

    # Thresholds (only if not debug)
    if not debug:
        filtered = []
        for c in dtw_candidates:
            if c['sim'] >= threshold_dtw:
                filtered.append(('dtw', c['start_s'], c['sim']))
        for c in corr_candidates:
            if c['r_mapped'] >= threshold_corr:
                filtered.append(('corr', c['start_s'], c['r_mapped']))
        filtered = sorted(filtered, key=lambda x: -x[2])[:top_k]
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'start_s', 'score'])
            for method, start_s, score in filtered:
                writer.writerow([method, start_s, score])

    # Short summary
    print(
        f"Found {len(dtw_sorted) + len(corr_sorted)} candidates (highest DTW={dtw_sorted[0]['sim']:.2f} at {dtw_sorted[0]['start_s']:.1f}s, highest corr={corr_sorted[0]['r_mapped']:.2f} at {corr_sorted[0]['start_s']:.1f}s).")
