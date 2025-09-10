
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
    # Use a custom threshold for positive class
    threshold = 0.3
    y_proba = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average='binary')
    cm = confusion_matrix(y_val, y_pred)
    print("Validation results:")
    print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print("Confusion matrix:\n", cm)

    joblib.dump(clf, args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == '__main__':
    main()
