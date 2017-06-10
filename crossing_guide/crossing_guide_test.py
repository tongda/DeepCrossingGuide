import unittest
from pathlib import Path
import numpy as np
import csv
from multiprocessing.pool import Pool

from .crossing_guide import CrossingGuide, CrossingMetrics, BatchIterator
from .util import read_image


def is_image_flipped(image1, image2):
    return np.array_equal(np.fliplr(image1), image2)


def is_metric_flipped(metric1, metric2):
    return metric1[1] == -metric2[1]


class BatchItoratorTest(unittest.TestCase):
    def test_should_return_batches(self):
        data_dir = "./data"
        piece_file = "./crossing_guide/test-piece.csv"

        with open(piece_file, 'r') as f:
            reader = csv.reader(f)
            raw_metrics = [CrossingMetrics(row) for row in reader]

        with Pool(5) as pool:
            iterator = BatchIterator(data_dir, raw_metrics, 2, pool)
            images, metrics = next(iterator)

        root = Path(data_dir)
        image_mapping = {metric.origin_metrics[0]: read_image(
            next(root.rglob("{}.jpg".format(metric.timestamp)))) for metric in raw_metrics}

        # check size
        self.assertEqual(images.shape, (2, 352, 288, 3))
        self.assertEqual(metrics.shape, (2, 12))
        # check order
        for met, img in zip(metrics, images):
            self.assertTrue(np.array_equal(image_mapping[met[0]], img))

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
        np.random.seed(10)
        guide.load_data(need_shuffle=False)

        with open(piece_file, 'r') as f:
            reader = csv.reader(f)
            metrics = [CrossingMetrics(row) for row in reader]

        root = Path(data_dir)
        images = [read_image(
            next(root.rglob("{}.jpg".format(metric.timestamp)))) for metric in metrics]
        train_datagen = guide._train_data_generator
        for i in range(10):
            batch = next(train_datagen)
            self.assertEqual(is_image_flipped(images[0], batch[0][0]), is_metric_flipped(
                metrics[0].origin_metrics, batch[1][0]),
                "No. {} loop failed.".format(i))
            self.assertEqual(is_image_flipped(images[1], batch[0][1]), is_metric_flipped(
                metrics[1].origin_metrics, batch[1][1]),
                "No. {} loop failed.".format(i))
