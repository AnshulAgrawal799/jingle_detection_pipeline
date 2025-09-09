#!/usr/bin/env bash
python -m jingle_detector \
  --jingle jingle_audio/wow_jingle.mp3 \
  --targets audio/wow_chunk_000.mp3 audio/rec1_chunk_000.mp3 \
  --plot_dir output/plots_debug \
  --output output/detections_debug.csv \
  --debug --top_k 10 --reencode-bad-mp3 --threshold-dtw 0.3 --threshold-corr 0.3
