# Jingle Detection Pipeline: Current Status (as of 2025-09-10)

## Workspace Overview

- **Workspace:** `jingle_analysis_pipeline`
- **Audio Data:**
  - Located in `audio/`
  - Files starting with `wow_` contain jingles (positive class)
  - Files starting with `rec1_` do **not** contain jingles (negative class)
- **Detection Candidates:**
  - `output/detections.csv` contains detection windows (filename, method, rank, start_s, score)
- **Waveform Plots:**
  - Plots with detection overlays for both wow and rec1 files

## Problem Statement

- **Issue:** Detection algorithm produces similar scores for both wow and rec1 files, causing false positives on rec1 (should be negative)
- **Goal:** Improve jingle detection accuracy, maximize F1, reduce rec1 false positives, maintain recall for wow

## Solution Approach

1. **Data Preparation**
   - Extract 1.5s windows centered on candidate start times from CSV
   - Additionally, extract multiple 1.5s sliding windows from `jingle_audio/wow_jingle.mp3` as positive (wow) training samples
   - Label windows as positive (wow) or negative (rec1) using filename
   - For each rec1 file, sample 20 extra negative windows (not overlapping with candidates)
   - Split by file: 80% train, 20% validation
2. **Feature Extraction**
   - **Primary:** Log-mel spectrograms (n_mels=64, n_fft=2048, hop_length=512, dB scale)
   - **Secondary (tabular):** MFCCs, short-time energy, spectral flux, centroid, rolloff, zero-crossing rate (mean/std)
   - Normalize tabular features using train set mean/std
3. **Model Candidates**
   - (A) Small 2D CNN on log-mel spectrograms
   - (B) LightGBM/XGBoost on tabular features (baseline)
4. **Training & Augmentation**
   - Augment with time stretch, pitch shift, noise, random gain
   - Adam optimizer, early stopping on val F1
   - Track precision, recall, F1, ROC AUC, PR AUC
5. **Hard Negative Mining**
   - Score rec1 windows, collect high-confidence false positives, add as hard negatives, retrain
6. **Threshold Calibration & Post-processing**
   - Find optimal threshold for F1
   - Merge detections within 0.5s, discard short detections, filter by energy/SNR
7. **Evaluation & Outputs**
   - Confusion matrix, precision/recall/F1, ROC/PR curves, per-file summary, updated CSV, model checkpoints, report

## What’s Working

- Loads audio and candidate CSV
- Extracts windows, labels, metadata
- Extracts additional positive training windows from `jingle_audio/wow_jingle.mp3` using a sliding window approach
- Computes log-mel and tabular features
- Normalizes tabular features
- Saves train/val splits, features, metadata
- Prints summary statistics and README
  **Data Preparation & Feature Extraction** (`prepare_features.py`):
- Loads audio and candidate CSV
- Extracts windows, labels, metadata
- 20 positive training windows from `jingle_audio/wow_jingle.mp3` have been added to `features/train_meta.csv` for improved model learning
- Computes log-mel and tabular features
- Normalizes tabular features
- Saves train/val splits, features, metadata
- Prints summary statistics and README
- **Baseline Training Stub** (`train_baseline.py`):
  - Loads features
  - Trains simple classifier (sklearn GradientBoosting as stub)
  - Outputs validation metrics, saves model
- **Requirements:** All necessary Python packages listed in `requirements.txt`

## What’s Next / To Do

- [ ] Regenerate features for train/val with new positives:
  ```bash
  python jingle_detector/prepare_features.py --audio_dir ./audio --csv_path ./features/train_meta.csv --out_dir ./features
  python jingle_detector/prepare_features.py --audio_dir ./audio --csv_path ./features/val_meta.csv --out_dir ./features
  ```
- [ ] Retrain baseline and CNN models:
  ```bash
  python jingle_detector/train_baseline.py --features_dir ./features --model_out ./models/baseline_model.joblib
  python jingle_detector/train_cnn.py --features_dir ./features --model_out ./models/cnn_model.pt
  ```
- [ ] Perform hard negative mining and threshold calibration
- [ ] Generate evaluation plots and reports

## Example Commands

```bash
python jingle_detector/prepare_features.py --audio_dir ./audio --csv_path ./features/train_meta.csv --out_dir ./features
python jingle_detector/prepare_features.py --audio_dir ./audio --csv_path ./features/val_meta.csv --out_dir ./features
python jingle_detector/train_baseline.py --features_dir ./features --model_out ./models/baseline_model.joblib
python jingle_detector/train_cnn.py --features_dir ./features --model_out ./models/cnn_model.pt
```

## Summary

You have a reproducible pipeline for extracting features and training a baseline jingle detector using filename-based weak labels. Next steps: run scripts, analyze results, iterate on model improvements.
