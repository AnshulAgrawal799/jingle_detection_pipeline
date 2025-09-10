import unittest
import numpy as np
from evaluate_binary import evaluate


def make_labels_scores():
    labels = {'a': 1, 'b': 0, 'c': 1, 'd': 0}
    scores = {'a': 0.9, 'b': 0.2, 'c': 0.7, 'd': 0.1}
    return labels, scores


class TestScoreAggregationAndThresholding(unittest.TestCase):
    def test_f1_at_threshold(self):
        labels, scores = make_labels_scores()
        results, roc_auc, y_true, y_score = evaluate(
            labels, scores, thresholds=[0.5])
        r = results[0]
        self.assertAlmostEqual(r['precision'], 1.0)
        self.assertAlmostEqual(r['recall'], 1.0)
        self.assertAlmostEqual(r['f1'], 1.0)
        self.assertEqual(r['cm'].tolist(), [[2, 0], [0, 2]])

    def test_threshold_sweep(self):
        labels, scores = make_labels_scores()
        results, roc_auc, y_true, y_score = evaluate(
            labels, scores, thresholds=[0.8, 0.5, 0.3])
        # At 0.8, only 'a' is positive
        self.assertEqual(results[0]['cm'].tolist(), [[2, 0], [1, 1]])
        # At 0.3, both 'a' and 'c' are positive
        self.assertEqual(results[2]['cm'].tolist(), [[2, 0], [0, 2]])

    def test_roc_auc(self):
        labels, scores = make_labels_scores()
        results, roc_auc, y_true, y_score = evaluate(labels, scores)
        self.assertAlmostEqual(roc_auc, 1.0)


if __name__ == '__main__':
    unittest.main()
