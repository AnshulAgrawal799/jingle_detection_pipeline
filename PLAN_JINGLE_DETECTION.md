# PLAN_JINGLE_DETECTION.md

## Goal

Develop a reliable method to automatically detect whether an audio file contains a jingle, with clear binary output ("present"/"absent") and validated performance.

---

## Assumptions and Dependencies

- Access to a set of audio files with ground-truth labels (jingle present/absent).
- Sufficient compute to train and evaluate models (CPU/GPU).
- Existing code for feature extraction and model training (see `jingle_detector/`).

---

## Required Datasets and Labeling Guidance

- **Format:** CSV with columns: `filename`, `label` (1 = jingle present, 0 = absent).
- **Sample size:** Minimum 100–200 labeled examples (balanced if possible).
- **Labeling:** Listen to each audio and mark if the jingle is clearly present anywhere.

---

## Evaluation Metrics

- Precision, recall, F1-score (primary).
- ROC AUC for threshold selection.
- Report confusion matrix.
- Test a range of thresholds (e.g., 0.3–0.7) to find optimal operating point.

---

## Prioritized Experiments

1. **Minimal PoC (small):**

   - Use existing CNN model (`train_cnn.py`, `cnn_model.pt`) and a tiny labeled set (e.g., 10 positive, 10 negative).
   - Run inference, plot score histograms, and compute metrics.

2. **Threshold tuning (small):**

   - Sweep thresholds, plot ROC/PR curves, select best threshold.

3. **Scale-up (medium):**

   - Label more data, retrain/fine-tune CNN, re-evaluate.

4. **Robustness tests (medium):**
   - Test on new/unseen audio, noisy backgrounds, different jingle variants.

---

## Minimal PoC Steps

1. **Prepare a tiny labeled CSV:**

   ```csv
   filename,label
   audio/wow_chunk_000.mp3,1
   audio/chunk_003.mp3,0
   ...
   ```

2. **Run inference:**

   ```powershell
   python -m jingle_detector --jingle jingle_audio/wow_jingle.mp3 --targets audio/wow_chunk_000.mp3 audio/chunk_003.mp3 --output output/detections.csv --window_step_s 2.5
   ```

3. **Post-process:**
   - For each file, take the max score as the detection score.
   - Compare to label, compute precision/recall.

---

## Code Changes Required

- Add a script for binary evaluation (e.g., `evaluate_binary.py`).
- Update `train_cnn.py` to support binary classification and thresholding.
- Example patch:
  ```python
  # ...existing code...
  # After running detection, aggregate max score per file and compare to label
  # Compute precision, recall, F1, ROC AUC
  ```
- Add unit tests for score aggregation and thresholding logic.

---

## Risk Assessment and Fallback Options

- **Risks:** Insufficient labeled data, ambiguous cases, overfitting to small datasets.
- **Fallbacks:** Use human-in-the-loop review for ambiguous cases, semi-supervised learning, or simple threshold-based heuristics as a baseline.
