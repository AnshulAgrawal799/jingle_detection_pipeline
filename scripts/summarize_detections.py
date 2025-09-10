import csv
import sys
from collections import defaultdict

# Usage: python summarize_detections.py input.csv output_summary.csv
if len(sys.argv) != 3:
    print("Usage: python summarize_detections.py input.csv output_summary.csv")
    sys.exit(1)

input_csv = sys.argv[1]
output_csv = sys.argv[2]

# Read all rows, keep max score per file (any method)
max_scores = {}

with open(input_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['filename']
        try:
            score = float(row['score'])
        except (ValueError, KeyError):
            continue
        if (fname not in max_scores) or (score > max_scores[fname][1]):
            max_scores[fname] = (row['method'], score, row['start_s'])

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'method', 'start_s', 'max_score'])
    for fname, (method, score, start_s) in max_scores.items():
        writer.writerow([fname, method, start_s, score])
