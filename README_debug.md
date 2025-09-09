# Jingle Detection Pipeline Debug Mode

## Debug Mode Output

- Prints sample rate, audio length, template length, feature shapes.
- Shows top-10 DTW and cross-correlation candidates (raw, normalized, mapped, timestamps).
- Reports global maxima for each method and their timestamps.
- Writes top-K candidate snippets to `output/snippets/`.
- In debug mode, thresholds are not applied; all top-K candidates are shown in CSV.

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

See `scripts/debug_run.sh` for a sample debug invocation.
