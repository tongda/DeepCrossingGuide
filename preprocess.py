import csv
import argparse
from itertools import chain
from pathlib import Path

import numpy as np
from scipy import signal

from util import read_metrics

pieces = [
    (149548966594, 149548968508),
    (149548979485, 149548980446),
    (149548990624, 149548992008),
    (149549001955, 149549003385),
    (149549010527, 149549011624),
    (149549040783, 149549041642),
    (149553054319, 149553055178),
    (149557630619, 149557632044),
    (149557643820, 149557644763),
    (149557654211, 149557655868),
    (149557674346, 149557675390)
]


def lpf(sig):
    b, a = signal.butter(3, 0.05, 'low')
    return signal.filtfilt(b, a, sig)


def process_piece(root, piece_no, start, end, writer):
    timestamps = [p.stem for p in root.glob("*.jpg")
                  if int(p.stem) >= start and int(p.stem) <= end]
    metrics = np.array([read_metrics(root / "{}.bin".format(ts))
                        for ts in timestamps])
    reset_metrics = metrics - metrics[0]
    filtered_metrics = np.apply_along_axis(lpf, 0, reset_metrics)
    for i in range(metrics.shape[0]):
        writer.writerow([piece_no, timestamps[i], *metrics[i].tolist(),
                         *reset_metrics[i].tolist(), *filtered_metrics[i].tolist()])


def preprocess(root: Path):
    with open("processed.csv", "w") as f:
        writer = csv.writer(f)
        for i, (start, end) in enumerate(pieces):
            process_piece(root, i, start, end, writer)


def main():
    parser = argparse.ArgumentParser(description='Preprocess Data.')
    parser.add_argument('--root', dest='root',
                        required=True, help='root path of data')
    parser.add_argument('--output', dest='output', default='processed.csv',
                        required=False, help='output path of processed data')
    args = parser.parse_args()
    root = Path(args.root)
    preprocess(root)


if __name__ == "__main__":
    main()
