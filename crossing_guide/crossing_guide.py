import csv
import functools
import logging
import random
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import activations
from keras.callbacks import Callback, ProgbarLogger, TensorBoard
from keras.layers import (BatchNormalization, Conv2D, Cropping2D, Dense,
                          Dropout, Flatten, Lambda, MaxPooling2D)
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .util import read_image, read_metrics


def feat_size(all_feat=True):
    if all_feat:
        return 12
    else:
        return 3


class CrossingMetrics(object):
    def __init__(self, row, all_feat=True):
        self.track = int(row[0])
        self.timestamp = int(row[1])
        self.origin_metrics = list(map(float, row[2:2 + feat_size(all_feat)]))
        self.reset_metrics = list(map(float, row[14:14 + feat_size(all_feat)]))
        self.filtered_metrics = list(
            map(float, row[26:26 + feat_size(all_feat)]))


class threadsafe_iter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    @functools.wraps(f)
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class CrossingGuide(object):
    def __init__(self, **conf):
        self.dropout_rate = conf.get("dropout_rate", 0.2)
        self.data_dir = conf.get('data_dir', './data/0524')
        self.activation = activations.get(conf.get('activation', 'relu'))
        self.batch_size = conf.get('batch_size', 128)
        self.use_lpf = conf.get("use_lpf", True)
        self.save_path = conf.get("save_path", "model.h5")
        self.all_feat = conf.get("all_feat", True)
        self.piece_file = conf.get("piece_file", "processed.csv")
        self.valid_ratio = conf.get("valid_ratio", 0.2)

        self.image_shape = conf.get('image_shape', (352, 288, 3))

        logging.info("Batch Size: {}".format(self.batch_size))

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255.0) -
                         0.5, input_shape=self.image_shape))
        model.add(Conv2D(16, (1, 1), padding='same',
                         activation=self.activation, kernel_initializer='glorot_normal'))
        model.add(Conv2D(32, (5, 5), padding='same',
                         activation=self.activation, kernel_initializer='glorot_normal'))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (5, 5), padding='same',
                         activation=self.activation, kernel_initializer='glorot_normal'))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (5, 5), padding='same',
                         activation=self.activation, kernel_initializer='glorot_normal'))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (3, 3), padding='same',
                         activation=self.activation, kernel_initializer='glorot_normal'))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, (3, 3), padding='same',
                         activation=self.activation, kernel_initializer='glorot_normal'))
        model.add(MaxPooling2D())
        model.add(Dropout(self.dropout_rate))
        model.add(Conv2D(128, (11, 9), padding='valid',
                         activation=self.activation, kernel_initializer='glorot_normal'))
        model.add(Dropout(self.dropout_rate))
        model.add(Conv2D(64, (1, 1), padding='valid',
                         activation=self.activation, kernel_initializer='glorot_normal'))
        model.add(Dropout(self.dropout_rate))
        model.add(Conv2D(16, (1, 1), padding='valid',
                         activation=self.activation, kernel_initializer='glorot_normal'))
        model.add(Conv2D(feat_size(self.all_feat), (1, 1), padding='valid',
                         activation=self.activation, kernel_initializer='glorot_normal'))
        model.add(Flatten())
        # model.add(Flatten())
        # model.add(Dense(128, activation=self.activation))
        # model.add(Dropout(self.dropout_rate))
        # model.add(Dense(64, activation=self.activation))
        # model.add(Dropout(self.dropout_rate))
        # model.add(Dense(16, activation=self.activation))
        # model.add(Dense(feat_size(self.all_feat)))

        model.compile(loss='mse', optimizer='adam')

        return model

    def load_data(self, need_shuffle=True):
        root = Path(self.data_dir)
        with open(self.piece_file, "r") as f:
            reader = csv.reader(f)
            metrics = [CrossingMetrics(row, self.all_feat) for row in reader]

        logging.info("{} sample found.".format(len(metrics)))
        ts_train, ts_valid = train_test_split(
            metrics, test_size=self.valid_ratio)

        @threadsafe_generator
        def generator(samples):
            while True:
                if need_shuffle:
                    shuffle(samples)
                for offset in range(0, len(samples), self.batch_size):
                    batch_metrics = samples[offset:offset + self.batch_size]
                    flips = [random.random() > 0.5 for _ in range(
                        len(batch_metrics))]
                    metrics_flip_array = np.ones(
                        (len(batch_metrics), feat_size(all_feat=self.all_feat)))
                    metrics_flip_array[:, 1] = np.array(flips) * 2 - 1
                    images = np.array(
                        [read_image(
                            next(root.rglob("{}.jpg".format(metric.timestamp))),
                            flip=flip
                        ) for metric, flip in zip(batch_metrics, flips)]
                    )
                    metrics = np.array(
                        [metric.filtered_metrics if self.use_lpf else metric.origin_metrics
                         for metric in batch_metrics]
                    )
                    metrics *= metrics_flip_array
                    yield images, metrics
        self._train_data_generator = generator(ts_train)
        self._train_size = len(ts_train)
        self._valid_data_generator = generator(ts_valid)
        self._valid_size = len(ts_valid)

    def train(self, num_epoch):
        self.load_data()
        logname = '-'.join(['dr' + str(self.dropout_rate),
                            'lpf' + str(self.use_lpf),
                            'ep' + str(num_epoch), activations.serialize(self.activation)])

        tfboard = TensorBoard('./logs/' + logname,
                              histogram_freq=1, write_graph=True)
        self.model.fit_generator(
            self._train_data_generator, steps_per_epoch=self._train_size / self.batch_size,
            validation_data=self._valid_data_generator,
            validation_steps=self._valid_size / self.batch_size,
            epochs=num_epoch, verbose=True,
            workers=10,
            callbacks=[tfboard],
            max_q_size=200)

        self.model.save(self.save_path)

    def load(self):
        self.model = load_model(self.save_path)

    def predict(self, image):
        return self.model.predict(np.expand_dims(image, 0), batch_size=1)
