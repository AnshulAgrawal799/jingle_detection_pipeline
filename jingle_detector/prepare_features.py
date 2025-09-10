import os
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from joblib import Parallel, delayed, dump
from sklearn.model_selection import train_test_split

# Constants
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--window_len_s', type=float, default=1.5)
    parser.add_argument('--extra_neg_per_file', type=int, default=20)
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--mfcc_n', type=int, default=13)
    return parser.parse_args()


def safe_load_audio(path, sr):
    try:
        y, file_sr = sf.read(path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        return y
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}")
        return None


def extract_window(y, center_s, sr, win_len_s):
    win_len = int(win_len_s * sr)
    center = int(center_s * sr)
    start = center - win_len // 2
    end = start + win_len
    pad_left = max(0, -start)
    pad_right = max(0, end - len(y))
    start = max(0, start)
    end = min(len(y), end)
    window = y[start:end]
    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, (pad_left, pad_right), mode='constant')
    return window


def compute_tabular_features(y, sr, n_mfcc=13):
    feats = []
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))
    # Delta and delta-delta MFCCs
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    feats.extend(np.mean(mfcc_delta, axis=1))
    feats.extend(np.std(mfcc_delta, axis=1))
    feats.extend(np.mean(mfcc_delta2, axis=1))
    feats.extend(np.std(mfcc_delta2, axis=1))
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats.extend(np.mean(contrast, axis=1))
    feats.extend(np.std(contrast, axis=1))
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(np.mean(chroma, axis=1))
    feats.extend(np.std(chroma, axis=1))
    # Short-time energy
    frame_len = 2048
    hop_len = 512
    energy = librosa.feature.rms(
        y=y, frame_length=frame_len, hop_length=hop_len)[0]
    feats.append(np.mean(energy))
    feats.append(np.std(energy))
    # Spectral flux
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    feats.append(np.mean(flux))
    feats.append(np.std(flux))
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats.append(np.mean(centroid))
    feats.append(np.std(centroid))
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    feats.append(np.mean(rolloff))
    feats.append(np.std(rolloff))
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_len, hop_length=hop_len)
    feats.append(np.mean(zcr))
    feats.append(np.std(zcr))
    return np.array(feats, dtype=np.float32)


def compute_mel(y, sr, n_mels, n_fft, hop_length, win_len_s):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Ensure fixed time frames
    expected_frames = int(np.ceil(win_len_s * sr / hop_length))
    if S_db.shape[1] < expected_frames:
        pad_width = expected_frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
    elif S_db.shape[1] > expected_frames:
        S_db = S_db[:, :expected_frames]
    # Normalize per-window
    S_db = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-6)
    return S_db.astype(np.float32)


def get_candidate_windows(csv_path):
    df = pd.read_csv(csv_path)
    candidates = {}
    for _, row in df.iterrows():
        fname = row['filename']
        start_s = float(row['start_s'])
        candidates.setdefault(fname, []).append(start_s)
    return candidates


def get_audio_files(audio_dir):
    return [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]


def sample_random_negatives(rec1_file, candidate_starts, y, sr, win_len_s, extra_neg_per_file):
    duration = len(y) / sr
    candidate_ranges = [(max(0, s - win_len_s/2 - 0.5), min(duration,
                         s + win_len_s/2 + 0.5)) for s in candidate_starts]
    neg_starts = []
    tries = 0
    while len(neg_starts) < extra_neg_per_file and tries < extra_neg_per_file * 10:
        rand_s = np.random.uniform(win_len_s/2, duration - win_len_s/2)
        overlap = any([low <= rand_s <= high for (
            low, high) in candidate_ranges])
        if not overlap:
            neg_starts.append(rand_s)
        tries += 1
    return neg_starts


def process_window(args):
    y, sr, start_s, win_len_s, n_mels, n_fft, hop_length, mfcc_n = args['y'], args['sr'], args[
        'start_s'], args['win_len_s'], args['n_mels'], args['n_fft'], args['hop_length'], args['mfcc_n']
    mel = compute_mel(y, sr, n_mels, n_fft, hop_length, win_len_s)
    tab = compute_tabular_features(y, sr, mfcc_n)
    return mel, tab


def main():
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'features'), exist_ok=True)

    audio_files = get_audio_files(args.audio_dir)
    candidates = get_candidate_windows(args.csv_path)
    meta_rows = []
    windows = []

    # --- Add positive sliding windows from wow_jingle.mp3 ---
    wow_jingle_path = os.path.join(os.path.dirname(
        args.audio_dir), 'jingle_audio', 'wow_jingle.mp3')
    if os.path.exists(wow_jingle_path):
        y_jingle = safe_load_audio(wow_jingle_path, args.sample_rate)
        if y_jingle is not None:
            win_len = int(args.window_len_s * args.sample_rate)
            hop_len = int(win_len // 2)  # 50% overlap
            total_len = len(y_jingle)
            n_windows = max(1, (total_len - win_len) // hop_len + 1)
            for i in range(n_windows):
                start = i * hop_len
                center_s = (start + win_len // 2) / args.sample_rate
                win = extract_window(y_jingle, center_s,
                                     args.sample_rate, args.window_len_s)
                windows.append({'y': win, 'sr': args.sample_rate, 'start_s': center_s,
                               'filename': 'wow_jingle.mp3', 'label': 1, 'source': 'jingle_audio'})
                meta_rows.append(
                    {'filename': 'wow_jingle.mp3', 'start_s': center_s, 'label': 1, 'source': 'jingle_audio'})
        else:
            print(
                f"Warning: Could not load {wow_jingle_path} for positive windows.")
    else:
        print(
            f"Warning: {wow_jingle_path} not found. No extra positive windows will be added.")

    print("Preparing candidate windows...")
    for fname in tqdm(audio_files):
        path = os.path.join(args.audio_dir, fname)
        y = safe_load_audio(path, args.sample_rate)
        if y is None:
            continue
        label = 1 if fname.startswith('wow') else 0
        candidate_starts = candidates.get(fname, [])
        # Candidate windows
        for start_s in candidate_starts:
            win = extract_window(
                y, start_s, args.sample_rate, args.window_len_s)
            windows.append({'y': win, 'sr': args.sample_rate, 'start_s': start_s,
                           'filename': fname, 'label': label, 'source': 'candidate'})
            meta_rows.append(
                {'filename': fname, 'start_s': start_s, 'label': label, 'source': 'candidate'})
        # Extra negatives for rec1
        if fname.startswith('rec1'):
            neg_starts = sample_random_negatives(
                fname, candidate_starts, y, args.sample_rate, args.window_len_s, args.extra_neg_per_file)
            for start_s in neg_starts:
                win = extract_window(
                    y, start_s, args.sample_rate, args.window_len_s)
                windows.append({'y': win, 'sr': args.sample_rate, 'start_s': start_s,
                               'filename': fname, 'label': 0, 'source': 'random_neg'})
                meta_rows.append(
                    {'filename': fname, 'start_s': start_s, 'label': 0, 'source': 'random_neg'})

    meta_df = pd.DataFrame(meta_rows)
    # Improved file-wise split: guarantee at least one wow and one rec1 in both train and val
    unique_files = meta_df['filename'].unique()
    wow_files = [f for f in unique_files if f.startswith('wow')]
    rec1_files = [f for f in unique_files if f.startswith('rec1')]
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(wow_files)
    np.random.shuffle(rec1_files)
    # Always at least one wow and one rec1 in both splits
    n_val_wow = min(max(1, int(0.2 * len(wow_files))),
                    len(wow_files)-1 if len(wow_files) > 1 else 1)
    n_val_rec1 = min(max(1, int(0.2 * len(rec1_files))),
                     len(rec1_files)-1 if len(rec1_files) > 1 else 1)
    n_train_wow = len(wow_files) - n_val_wow
    n_train_rec1 = len(rec1_files) - n_val_rec1
    train_files = list(wow_files[:n_train_wow]) + \
        list(rec1_files[:n_train_rec1])
    val_files = list(wow_files[n_train_wow:]) + list(rec1_files[n_train_rec1:])
    # If there are files that are neither wow nor rec1, add them to train
    other_files = [f for f in unique_files if not (
        f.startswith('wow') or f.startswith('rec1'))]
    train_files += other_files
    train_idx = meta_df['filename'].isin(train_files)
    val_idx = meta_df['filename'].isin(val_files)
    train_meta = meta_df[train_idx].reset_index(drop=True)
    val_meta = meta_df[val_idx].reset_index(drop=True)

    # Feature extraction
    print("Extracting features (this may take a while)...")

    def extract_feats(row):
        return process_window({
            'y': row['y'],
            'sr': row['sr'],
            'start_s': row['start_s'],
            'win_len_s': args.window_len_s,
            'n_mels': args.n_mels,
            'n_fft': args.n_fft,
            'hop_length': args.hop_length,
            'mfcc_n': args.mfcc_n
        })

    train_windows = [w for i, w in enumerate(windows) if train_idx[i]]
    val_windows = [w for i, w in enumerate(windows) if val_idx[i]]

    train_feats = Parallel(n_jobs=-1)(delayed(extract_feats)(w)
                                      for w in tqdm(train_windows))
    val_feats = Parallel(n_jobs=-1)(delayed(extract_feats)(w)
                                    for w in tqdm(val_windows))

    X_mel_train = np.stack([f[0] for f in train_feats])
    X_tab_train = np.stack([f[1] for f in train_feats])
    y_train = train_meta['label'].values.astype(np.int8)

    X_mel_val = np.stack([f[0] for f in val_feats])
    X_tab_val = np.stack([f[1] for f in val_feats])
    y_val = val_meta['label'].values.astype(np.int8)

    # Tabular normalization
    tab_mean = X_tab_train.mean(axis=0)
    tab_std = X_tab_train.std(axis=0) + 1e-6
    X_tab_train_norm = (X_tab_train - tab_mean) / tab_std
    X_tab_val_norm = (X_tab_val - tab_mean) / tab_std
    feature_stats = {'mean': tab_mean, 'std': tab_std}
    dump(feature_stats, os.path.join(args.out_dir, 'feature_stats.joblib'))

    # Save outputs
    np.savez(os.path.join(args.out_dir, 'train_features.npz'),
             X_mel=X_mel_train, X_tab=X_tab_train_norm, y=y_train)
    np.savez(os.path.join(args.out_dir, 'val_features.npz'),
             X_mel=X_mel_val, X_tab=X_tab_val_norm, y=y_val)
    train_meta.to_csv(os.path.join(
        args.out_dir, 'train_meta.csv'), index=False)
    val_meta.to_csv(os.path.join(args.out_dir, 'val_meta.csv'), index=False)

    # Print summary
    print("\nSummary:")
    print(
        f"Train windows: {len(y_train)} (pos={np.sum(y_train)}, neg={len(y_train)-np.sum(y_train)})")
    print(
        f"Val windows: {len(y_val)} (pos={np.sum(y_val)}, neg={len(y_val)-np.sum(y_val)})")
    print(f"Example mel shape: {X_mel_train[0].shape}")
    print(f"Tabular feature shape: {X_tab_train.shape}")
    print("Energy percentiles (train):")
    wow_energy = X_tab_train[y_train == 1, 26]  # energy mean
    rec1_energy = X_tab_train[y_train == 0, 26]
    if wow_energy.size > 0:
        print(f"  wow: {np.percentile(wow_energy, [0,25,50,75,100])}")
    else:
        print("  wow: (no positive samples in train)")
    if rec1_energy.size > 0:
        print(f"  rec1: {np.percentile(rec1_energy, [0,25,50,75,100])}")
    else:
        print("  rec1: (no negative samples in train)")

    # README
    print("\nREADME:")
    print("Feature files produced:")
    print("  ./features/train_features.npz, ./features/val_features.npz: arrays for model training")
    print("  ./features/train_meta.csv, ./features/val_meta.csv: metadata for each window")
    print("  ./features/feature_stats.joblib: normalization stats for tabular features")
    print("To train a baseline model, run:")
    print("  python train_baseline.py --features_dir ./features --model_out ./models/baseline_model.joblib")
    print('Train wow:', (train_meta['label'] == 1).sum(
    ), 'rec1:', (train_meta['label'] == 0).sum())
    print('Val wow:', (val_meta['label'] == 1).sum(),
          'rec1:', (val_meta['label'] == 0).sum())


if __name__ == '__main__':
    main()
