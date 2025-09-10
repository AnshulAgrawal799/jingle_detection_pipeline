"""
evaluate_binary.py

Script to evaluate binary jingle detection performance using model outputs and ground-truth labels.
- Computes precision, recall, F1, ROC AUC, confusion matrix, and threshold sweep plots.
- Aggregates per-file scores (e.g., max score per file) for binary decision.

Usage:
    python evaluate_binary.py --detections_csv output/detections.csv --labels_csv features/val_meta.csv --out_dir output/eval
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def load_labels(labels_csv):
    df = pd.read_csv(labels_csv)
    # Expect columns: filename, label
    return dict(zip(df['filename'], df['label']))


def load_detections(detections_csv):
    df = pd.read_csv(detections_csv)
    # Expect columns: filename, score (optionally: start, end, etc.)
    # Aggregate max score per file
    agg = df.groupby('filename')['score'].max().reset_index()
    return dict(zip(agg['filename'], agg['score']))


def evaluate(labels, scores, thresholds=np.linspace(0, 1, 101)):
    y_true = []
    y_score = []
    for fname in labels:
        if fname in scores:
            y_true.append(labels[fname])
            y_score.append(scores[fname])
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    results = []
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        results.append({'threshold': thresh, 'precision': p,
                       'recall': r, 'f1': f1, 'cm': cm})
    roc_auc = roc_auc_score(y_true, y_score)
    return results, roc_auc, y_true, y_score


def plot_curves(y_true, y_score, out_dir):
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
    plt.close()
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(out_dir, 'pr_curve.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detections_csv', required=True)
    parser.add_argument('--labels_csv', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    labels = load_labels(args.labels_csv)
    scores = load_detections(args.detections_csv)
    results, roc_auc, y_true, y_score = evaluate(labels, scores)
    # Save metrics
    metrics = pd.DataFrame(
        [{k: v for k, v in r.items() if k != 'cm'} for r in results])
    metrics.to_csv(os.path.join(
        args.out_dir, 'metrics_by_threshold.csv'), index=False)
    # Save best F1
    best = max(results, key=lambda r: r['f1'])
    with open(os.path.join(args.out_dir, 'best_f1.txt'), 'w') as f:
        f.write(
            f"Best F1: {best['f1']:.3f} at threshold {best['threshold']:.2f}\n")
        f.write(
            f"Precision: {best['precision']:.3f}, Recall: {best['recall']:.3f}\n")
        f.write(f"Confusion matrix:\n{best['cm']}\n")
        f.write(f"ROC AUC: {roc_auc:.3f}\n")
    # Plot curves
    plot_curves(y_true, y_score, args.out_dir)


if __name__ == '__main__':
    main()
