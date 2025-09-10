
import os
import argparse
import numpy as np
import joblib
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, required=True)
    parser.add_argument('--model_out', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    train = np.load(os.path.join(args.features_dir, 'train_features.npz'))
    val = np.load(os.path.join(args.features_dir, 'val_features.npz'))
    X_train, y_train = train['X_tab'], train['y']
    # Oversample positives in the training set
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    X_val, y_val = val['X_tab'], val['y']

    # LightGBM/XGBoost preferred, but using sklearn GradientBoosting for stub
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0
    best = None
    from sklearn.metrics import roc_auc_score
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best = dict(threshold=threshold, precision=prec,
                        recall=rec, f1=f1, cm=cm)
    roc_auc = roc_auc_score(y_val, y_proba)
    print("Validation results (best threshold sweep):")
    print(f"Best F1: {best['f1']:.3f} at threshold {best['threshold']:.2f}")
    print(f"Precision: {best['precision']:.3f}, Recall: {best['recall']:.3f}")
    print("Confusion matrix:\n", best['cm'])
    print(f"ROC AUC: {roc_auc:.3f}")
    # Optionally save per-window scores for later eval
    np.savez(os.path.join(os.path.dirname(args.model_out),
             'val_scores_baseline.npz'), y_val=y_val, y_proba=y_proba)
    joblib.dump(clf, args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == '__main__':
    main()
