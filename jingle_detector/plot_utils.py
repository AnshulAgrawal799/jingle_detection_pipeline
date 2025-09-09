import matplotlib.pyplot as plt
import numpy as np


def plot_detection(y, sr, detections, out_path, title):
    plt.figure(figsize=(12, 4))
    times = np.arange(len(y)) / sr
    plt.plot(times, y, label='Audio')
    # Only add legend for first detection to avoid duplicate labels
    legend_added = False
    for start, end, score in detections:
        if not legend_added:
            plt.axvspan(start, end, color='red', alpha=0.3,
                        label=f'Detected (score={score:.2f})')
            legend_added = True
        else:
            plt.axvspan(start, end, color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scores_vs_time(scores, sr, out_path, method_name, top_peaks=None):
    """
    Plot similarity/correlation scores vs time, annotate peaks/minima.
    """
    plt.figure(figsize=(12, 4))
    times = np.array([s[0] / sr for s in scores])
    sim_values = np.array([s[3] for s in scores])
    plt.plot(times, sim_values, label=f'{method_name} similarity')
    if top_peaks:
        for idx in top_peaks:
            plt.plot(times[idx], sim_values[idx], 'ro',
                     label='Top candidate' if idx == top_peaks[0] else None)
    plt.title(f'{method_name} similarity vs time')
    plt.xlabel('Time (s)')
    plt.ylabel('Similarity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_comparison(template_feat, match_feat, out_path, feature_type):
    """
    Compare template vs matched segment features (e.g., log-mel, chroma).
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(template_feat, aspect='auto', origin='lower')
    plt.title('Template ' + feature_type)
    plt.subplot(1, 2, 2)
    plt.imshow(match_feat, aspect='auto', origin='lower')
    plt.title('Matched ' + feature_type)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
