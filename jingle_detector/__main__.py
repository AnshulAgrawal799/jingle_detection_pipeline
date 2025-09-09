import argparse
import os
import csv
import logging
from .audio_utils import load_audio, extract_chroma, extract_logmel
from .detector import detect_dtw, detect_corr, DetectionResult
from .plot_utils import plot_detection
from .config import DEFAULT_SR, DTW_THRESH, CORR_THRESH, WIN_LEN, HOP_LENGTH, N_MELS

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    parser = argparse.ArgumentParser(description="Jingle detection pipeline")
    parser.add_argument('--jingle', required=True,
                        help='Path to jingle template audio')
    parser.add_argument('--targets', nargs='+',
                        required=True, help='Target audio files')
    parser.add_argument('--plot_dir', default='output/plots',
                        help='Directory for plots')
    parser.add_argument(
        '--output', default='output/detections.csv', help='Output CSV file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top candidates to keep')
    parser.add_argument('--reencode-bad-mp3',
                        action='store_true', help='Always re-encode MP3s')
    parser.add_argument('--threshold-dtw', type=float,
                        default=0.3, help='DTW similarity threshold')
    parser.add_argument('--threshold-corr', type=float,
                        default=0.3, help='Correlation threshold')
    parser.add_argument('--window_step_s', type=float,
                        default=0.1, help='Sliding window step (seconds)')
    parser.add_argument('--window_size_s', type=float,
                        default=None, help='Sliding window size (seconds)')
    args = parser.parse_args()

    # Handle missing sr gracefully
    if not hasattr(args, 'sr') or args.sr is None:
        args.sr = DEFAULT_SR

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    from . import detector
    detector.process_target(
        jingle_path=args.jingle,
        target_paths=args.targets,
        plot_dir=args.plot_dir,
        output_csv=args.output,
        debug=args.debug,
        top_k=args.top_k,
        reencode_bad_mp3=args.reencode_bad_mp3,
        threshold_dtw=args.threshold_dtw,
        threshold_corr=args.threshold_corr,
        window_step_s=args.window_step_s,
        window_size_s=args.window_size_s
    )
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    # Load jingle
    jingle_y, sr = load_audio(args.jingle, sr=args.sr)
    jingle_chroma = extract_chroma(jingle_y, sr)
    jingle_logmel = extract_logmel(jingle_y, sr, n_mels=args.n_mels)
    jingle_len = len(jingle_y) / sr
    if args.debug:
        print(f"[DEBUG] Jingle: sr={sr}, length={jingle_len:.2f}s")
        print(
            f"[DEBUG] Jingle chroma shape: {jingle_chroma.shape}, logmel shape: {jingle_logmel.shape}")

    all_results = []
    for target_path in args.targets:
        logging.info(f'Processing {target_path}...')
        y, sr = load_audio(target_path, sr=args.sr)
        target_len = len(y) / sr
        chroma = extract_chroma(y, sr)
        logmel = extract_logmel(y, sr, n_mels=args.n_mels)
        if args.debug:
            print(f"[DEBUG] Target: sr={sr}, length={target_len:.2f}s")
            print(
                f"[DEBUG] Target chroma shape: {chroma.shape}, logmel shape: {logmel.shape}")


if __name__ == '__main__':
    main()
