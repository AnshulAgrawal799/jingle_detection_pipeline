import argparse
import os
import csv
import logging
from .audio_utils import load_audio, extract_chroma, extract_logmel
from .detector import detect_dtw, detect_corr, DetectionResult
from .plot_utils import plot_detection
from .config import DEFAULT_SR, DTW_THRESH, CORR_THRESH, WIN_LEN, HOP_LEN, N_MELS

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    parser = argparse.ArgumentParser(description='Jingle detection tool')
    parser.add_argument('--jingle', required=True,
                        help='Path to reference jingle mp3')
    parser.add_argument('--targets', nargs='+', required=True,
                        help='Target audio files to scan')
    parser.add_argument('--output', required=True, help='CSV output path')
    parser.add_argument('--plot_dir', default=None,
                        help='Directory to save diagnostic plots')
    parser.add_argument('--dtw_thresh', type=float, default=DTW_THRESH)
    parser.add_argument('--corr_thresh', type=float, default=CORR_THRESH)
    parser.add_argument('--win_len', type=float, default=WIN_LEN)
    parser.add_argument('--hop_len', type=float, default=HOP_LEN)
    parser.add_argument('--sr', type=int, default=DEFAULT_SR)
    parser.add_argument('--n_mels', type=int, default=N_MELS)
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug diagnostics')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Export top-K candidate matches')
    parser.add_argument('--reencode-bad-mp3', action='store_true',
                        help='Re-encode bad MP3s to WAV if header warnings')
    parser.add_argument('--threshold-dtw', type=float,
                        default=DTW_THRESH, help='DTW similarity threshold')
    parser.add_argument('--threshold-corr', type=float,
                        default=CORR_THRESH, help='Correlation similarity threshold')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
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

        # DTW detection
        dtw_top, dtw_raw = detect_dtw(
            jingle_chroma, chroma, sr, args.win_len, args.hop_len, top_k=args.top_k, thresh=args.threshold_dtw, beta=1.0, debug=args.debug)
        # Correlation detection
        corr_top, corr_raw = detect_corr(
            jingle_logmel, logmel, sr, args.win_len, args.hop_len, top_k=args.top_k, thresh=args.threshold_corr, debug=args.debug)

        results = []
        for start, end, score in dtw_top:
            results.append(DetectionResult(os.path.basename(
                target_path), start, end, score, 'dtw'))
        for start, end, score in corr_top:
            results.append(DetectionResult(os.path.basename(
                target_path), start, end, score, 'correlation'))

        if args.plot_dir:
            plot_detection(y, sr, dtw_top + corr_top,
                           os.path.join(args.plot_dir, os.path.basename(
                               target_path) + '_detections.png'),
                           f'Detections in {os.path.basename(target_path)}')

        # Debug: print top 10 raw scores for each method
        if args.debug:
            print(
                f"[DEBUG] Top 10 DTW scores: {[round(x[3],3) for x in sorted(dtw_raw, key=lambda x: -x[3])[:10]]}")
            print(
                f"[DEBUG] Top 10 Corr scores: {[round(x[3],3) for x in sorted(corr_raw, key=lambda x: -x[3])[:10]]}")
            max_dtw = max(dtw_raw, key=lambda x: x[3])
            max_corr = max(corr_raw, key=lambda x: x[3])
            print(
                f"[DEBUG] Max DTW sim: {max_dtw[3]:.3f} at {max_dtw[0]/sr:.2f}s")
            print(
                f"[DEBUG] Max Corr sim: {max_corr[3]:.3f} at {max_corr[0]/sr:.2f}s")

        # Print summary per file
        if results:
            highest_dtw = max(
                [r.score for r in results if r.method == 'dtw'], default=0)
            highest_corr = max(
                [r.score for r in results if r.method == 'correlation'], default=0)
            logging.info(
                f"{len(results)} candidates found (highest score dtw={highest_dtw:.2f}, corr={highest_corr:.2f})")
        else:
            results.append(DetectionResult(os.path.basename(
                target_path), '-', '-', '-', 'no detection'))
            logging.info('No jingle detected.')
        all_results.extend(results)

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'start_s', 'end_s', 'score', 'method'])
        for r in all_results:
            writer.writerow(r.to_row())
    logging.info(f'CSV written to {args.output}')


if __name__ == '__main__':
    main()
