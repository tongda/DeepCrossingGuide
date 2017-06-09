import unittest
import random
from pathlib import Path
import numpy as np
import csv

from .crossing_guide import CrossingGuide, CrossingMetrics
from .util import read_image


def is_image_flipped(image1, image2):
    return np.array_equal(np.fliplr(image1), image2)

def is_metric_flipped(metric1, metric2):
    return metric1[1] == metric2[1]


class CrossingGuideTest(unittest.TestCase):
    def test_should_generate_fliped_images(self):
        data_dir = "./data"
        piece_file = "./crossing_guide/test-piece.csv"
        guide = CrossingGuide(data_dir=data_dir,
                              piece_file=piece_file,
                              batch_size=2,
                              valid_ratio=0,
                              use_lpf=False)
        # this produce [0.5714025946899135, 0.4288890546751146]
        random.seed(10)
        guide.load_data(need_shuffle=False)

        with open(piece_file, 'r') as f:
            reader = csv.reader(f)
            metrics = [CrossingMetrics(row) for row in reader]

        root = Path(data_dir)
        images = [read_image(
            next(root.rglob("{}.jpg".format(metric.timestamp)))) for metric in metrics]
        train_datagen = guide._train_data_generator
        batch = next(train_datagen)
        self.assertEqual(is_image_flipped(images[0], batch[0][0]), is_metric_flipped(metrics[0].origin_metrics, batch[1][0]))
        self.assertEqual(is_image_flipped(images[1], batch[1][0]), is_metric_flipped(metrics[1].origin_metrics, batch[1][1]))
