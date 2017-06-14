import argparse
import shutil
from multiprocessing import pool
from pathlib import Path

import numpy as np

from crossing_guide.util import read_image, read_metrics
from preprocess import pieces

SPANS = np.linspace(-np.pi / 2, np.pi / 2, 15)


def process_piece(root, piece_no, start, end, output_dir: Path, mode):
    timestamps = [p.stem for p in root.rglob("*.jpg")
                  if int(p.stem) >= start and int(p.stem) <= end]
    index_to_check = 0 if mode == 'landscape' else 1

    for ts in timestamps:
        met = read_metrics(next(root.rglob("{}.bin".format(ts))))
        cat_index = np.where(SPANS > met[index_to_check])[0][0]
        image_file = next(root.rglob("{}.jpg".format(ts)))

        if not (output_dir / str(cat_index)).exists():
            (output_dir / str(cat_index)).mkdir()

        shutil.copy(str(image_file), str(
            output_dir / str(cat_index) / "{}.jpg".format(ts)))

    print("processed: {}".format(piece_no))


def preprocess(root: Path, output_dir, mode: str):
    my_pool = pool.Pool(8)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    for i, (start, end) in enumerate(pieces):
        my_pool.apply_async(
            process_piece, (root, i, start, end, output_dir, mode))

    my_pool.close()
    my_pool.join()


def main():
    parser = argparse.ArgumentParser(description='Preprocess Data.')
    parser.add_argument('--root', dest='root',
                        required=True, help='root path of data')
    parser.add_argument('--output-dir', dest='output_dir', default='categorized',
                        required=False, help='output directory of processed data')
    parser.add_argument('--mode', dest='mode', default='landscape',
                        required=False, help='mode of processing: landscape or protrait')
    args = parser.parse_args()
    root = Path(args.root)
    preprocess(root, args.output_dir, args.mode)


if __name__ == "__main__":
    main()
