import matplotlib.pyplot as plt
import numpy as np


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


def plot_correlation_vs_time(r_mapped, sr, template_len_samples, outpath, top_k_indices=[]):
    times = np.arange(len(r_mapped)) / sr
    plt.figure(figsize=(10, 4))
    plt.plot(times, r_mapped, label='Correlation')
    for idx in top_k_indices:
        plt.axvline(times[idx], color='r', linestyle='--',
                    label=f'Top-{idx+1}' if idx == top_k_indices[0] else None)
    plt.xlabel('Time (s)')
    plt.ylabel('Mapped Correlation [0,1]')
    if top_k_indices:
        plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
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

            import matplotlib.pyplot as plt
            import numpy as np

            def plot_detection(y, sr, detections, out_path, title):
                plt.figure(figsize=(12, 4))
                times = np.arange(len(y)) / sr
                plt.plot(times, y, label='Audio')
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

            def plot_correlation_vs_time(r_mapped, sr, template_len_samples, outpath, top_k_indices=[]):
                times = np.arange(len(r_mapped)) / sr
                plt.figure(figsize=(10, 4))
                plt.plot(times, r_mapped, label='Correlation')
                for idx in top_k_indices:
                    plt.axvline(times[idx], color='r', linestyle='--',
                                label=f'Top-{idx+1}' if idx == top_k_indices[0] else None)
                plt.xlabel('Time (s)')
                plt.ylabel('Mapped Correlation [0,1]')
                if top_k_indices:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(outpath)
                plt.close()

            def plot_dtw_scores_vs_time(similarities, frame_indices, sr, hop_length, outpath, top_k_indices=[]):
                times = (np.array(frame_indices) * hop_length) / sr
                plt.figure(figsize=(10, 4))
                plt.plot(times, similarities, label='DTW Similarity')
                for idx in top_k_indices:
                    plt.axvline(times[idx], color='g', linestyle='--',
                                label=f'Top-{idx+1}' if idx == top_k_indices[0] else None)
                plt.xlabel('Time (s)')
                plt.ylabel('DTW Similarity [0,1]')
                if top_k_indices:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(outpath)
                plt.close()

            def plot_template_vs_match_spectrograms(template_logmel, match_logmel, outpath):
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(template_logmel, aspect='auto', origin='lower')
                plt.title('Template Log-Mel')
                plt.subplot(1, 2, 2)
                plt.imshow(match_logmel, aspect='auto', origin='lower')
                plt.title('Match Log-Mel')
                plt.tight_layout()
                plt.savefig(outpath)
                plt.close()
