# Jingle Detection Pipeline Debug Mode

## Important Limitation: No Reliable Binary Jingle Detection

**This repository does NOT provide a reliable, validated method to automatically distinguish between audios that contain a jingle and those that do not.**

- The current pipeline outputs top-K candidate segments with similarity scores (DTW/correlation), but does not make a binary decision ("jingle present" or "jingle absent").
- In debug mode, thresholds are not applied; all top-K candidates are shown, regardless of match quality.
- In non-debug mode, thresholds are user-defined and not calibrated for reliable detection. The score scales are arbitrary and not interpretable as probabilities.
- There is no recommended threshold or guidance for making presence/absence decisions based on the scores.
- As a result, **the outputs (including the attached CSVs) cannot be reliably used to decide if a jingle is present or absent in an audio file.**

**Users must interpret the results manually and with caution.**

## Debug Mode Output

- Prints sample rate, audio length, template length, feature shapes.
- Shows top-10 DTW and cross-correlation candidates (raw, normalized, mapped, timestamps).
- Reports global maxima for each method and their timestamps.
- Writes top-K candidate snippets to `output/snippets/`.
- **In debug mode, thresholds are not applied; all top-K candidates are shown in CSV, regardless of whether a jingle is actually present.**

## DTW Similarity Mapping

- Computes average per-step DTW distance between template and candidate window.
- Maps to similarity: `sim = exp(-beta * avg_dist)`.
- `beta` controls sensitivity: lower beta → higher similarities.
- Distances have no fixed scale; interpret mapped similarity as relative.

## Cross-Correlation Interpretation

- Normalized cross-correlation `r_norm` in [-1,1].
- Mapped to [0,1] using `(r+1)/2`.
- Higher values indicate better match.

## MP3 Re-encoding

- Use `--reencode-bad-mp3` to always re-encode MP3s to WAV for robust loading.
- Requires `ffmpeg` to be installed and available in PATH.

## Example Run

To run detection on a file and save results with the new naming convention:

```powershell
python -m jingle_detector --jingle jingle_audio/wow_jingle.mp3 --targets audio/wow_chunk_000.mp3 --output output/detections.csv --window_step_s 2.5
python -m jingle_detector --jingle jingle_audio/wow_jingle.mp3 --targets audio/rec1_chunk_000.mp3 --output output/detections.csv --window_step_s 2.5
python -m jingle_detector --jingle jingle_audio/wow_jingle.mp3 --targets audio/chunk_003.mp3 --output output/detections.csv --window_step_s 2.5
```

This will produce `output/detections_wow_chunk_000.csv`, `output/detections_rec1_chunk_000.csv`, and `output/detections_chunk_003.csv` respectively.
