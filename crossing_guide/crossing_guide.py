import csv
import datetime
import functools
import logging
import random
import threading
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import activations
from keras.activations import relu
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.callbacks import Callback, ProgbarLogger, TensorBoard
from keras.layers import (BatchNormalization, Conv2D, Cropping2D, Dense,
                          Dropout, Flatten, Lambda, MaxPooling2D)
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .util import read_image, read_metrics


def feat_size(all_feat=True):
    if all_feat:
        return 12
    else:
        return 2


class CrossingMetrics(object):
    def __init__(self, row, all_feat=True):
        self.track = int(row[0])
        self.timestamp = int(row[1])
        self.origin_metrics = list(map(float, row[2:2 + feat_size(all_feat)]))
        self.reset_metrics = list(map(float, row[14:14 + feat_size(all_feat)]))
        self.filtered_metrics = list(
            map(float, row[26:26 + feat_size(all_feat)]))


class ImageGenerator(object):
    def __init__(self,
                 root_dir,
                 use_lpf=False,
                 random_flip=False,
                 view_type="PORTRAIT"):
        self.root_dir = Path(root_dir)
        self.random_flip = random_flip
        self.use_lpf = use_lpf
        self.view_type = view_type

    def generate(self, raw_metric):
        image = read_image(next(self.root_dir.rglob(
            "{}.jpg".format(raw_metric.timestamp))))
        result_metric = raw_metric.filtered_metrics if self.use_lpf else raw_metric.origin_metrics
        if self.random_flip and random.random() > 0.5:
            image = np.fliplr(image)
            if self.view_type == "PORTRAIT":
                result_metric[1] *= -1
            else:
                result_metric[0] *= -1
        return image, result_metric


class BatchIterator(object):
    def __init__(self,
                 root_dir,
                 metrics,
                 batch_size,
                 worker_pool: Pool,
                 use_lpf=False,
                 random_flip=False,
                 view_type="PORTRAIT",
                 need_shuffle=True):
        self.batch_generator = self._flow(metrics, batch_size, need_shuffle)
        self.root_dir = Path(root_dir)
        self.random_flip = random_flip
        self.worker_pool = worker_pool
        self.use_lpf = use_lpf
        self.view_type = view_type
        self.generator = ImageGenerator(
            root_dir, use_lpf, random_flip, view_type)
        self.lock = threading.Lock()

    def _flow(self, samples, batch_size, need_shuffle):
        while True:
            if need_shuffle:
                shuffle(samples)
            for offset in range(0, len(samples), batch_size):
                batch_metrics = samples[offset:offset + batch_size]
                yield batch_metrics

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            batch_metrics = next(self.batch_generator)
        batch_pairs = self.worker_pool.map(
            self.generator.generate, batch_metrics)
        images, metrics = zip(*batch_pairs)
        return preprocess_input(np.array(images, dtype=np.float32)), np.array(metrics)


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
        self.all_feat = conf.get("all_feat", False)
        self.piece_file = conf.get("piece_file", "processed.csv")
        self.valid_ratio = conf.get("valid_ratio", 0.2)

        self.image_shape = conf.get('image_shape', (352, 288, 3))

        logging.info("Batch Size: {}".format(self.batch_size))

        self.model = self.build_model()
        self.worker_pool = Pool(processes=conf.get("process_pool_size", 4))

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
        model.add(Conv2D(128, (self.image_shape[0] // 32, self.image_shape[1] // 32), padding='valid',
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

        def create_iterator(dataset):
            return BatchIterator(
                self.data_dir,
                dataset,
                self.batch_size,
                self.worker_pool,
                self.use_lpf,
                random_flip=True,
                view_type="PORTRAIT",
                need_shuffle=need_shuffle)
        self._train_data_generator = create_iterator(ts_train)
        self._train_size = len(ts_train)
        self._valid_data_generator = create_iterator(ts_valid)
        self._valid_size = len(ts_valid)

    def train(self, num_epoch):
        self.load_data()
        logname = '-'.join(['dr' + str(self.dropout_rate),
                            'lpf' + str(self.use_lpf),
                            'ep' + str(num_epoch),
                            activations.serialize(self.activation),
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        tfboard = TensorBoard('./logs/' + logname,
                              histogram_freq=1, write_graph=True)
        self.model.fit_generator(
            self._train_data_generator, steps_per_epoch=self._train_size / self.batch_size,
            validation_data=self._valid_data_generator,
            validation_steps=self._valid_size / self.batch_size,
            epochs=num_epoch, verbose=True,
            workers=10,
            # class_weights={},
            callbacks=[tfboard],
            max_q_size=200)

        self.model.save(self.save_path)

    def load(self):
        self.model = load_model(self.save_path)

    def predict(self, image):
        return self.model.predict(np.expand_dims(image, 0), batch_size=1)


class CrossingGuideV2(CrossingGuide):
    def __init__(self, *args, **kwargs):
        super(CrossingGuideV2, self).__init__(*args, **kwargs)

    def build_model(self):
        vgg = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=self.image_shape
        )

        for layer in vgg.layers[:-4]:
            layer.trainable = False

        x = vgg.output

        x = Dropout(self.dropout_rate)(x)
        x = Conv2D(128, (self.image_shape[0] // 32, self.image_shape[1] // 32), padding='valid',
                   activation=self.activation, kernel_initializer='glorot_normal')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv2D(64, (1, 1), padding='valid',
                   activation=self.activation, kernel_initializer='glorot_normal')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv2D(feat_size(self.all_feat), (1, 1), padding='valid',
                   activation=None, kernel_initializer='glorot_normal')(x)
        x = Flatten()(x)

        model = Model(inputs=vgg.input, outputs=x)
        model.compile(loss='mse', optimizer='adam')

        return model


def preprocess_input(x):
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x


class CrossingGuideV3(CrossingGuide):
    def __init__(self, *args, **kwargs):
        super(CrossingGuideV3, self).__init__(*args, **kwargs)

    def build_model(self):
        vgg = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=self.image_shape
        )

        for layer in vgg.layers[:-4]:
            layer.trainable = False

        x = vgg.output

        lrelu = functools.partial(relu, alpha=0.1)
        lrelu.__name__ = 'lrelu'

        x = Dropout(self.dropout_rate)(x)
        x = Conv2D(128, (self.image_shape[0] // 32, self.image_shape[1] // 32), padding='valid',
                   activation=lrelu, kernel_initializer='glorot_normal')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv2D(64, (1, 1), padding='valid',
                   activation=lrelu, kernel_initializer='glorot_normal')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv2D(16, (1, 1), padding='valid',
                   activation='softmax', kernel_initializer='glorot_normal')(x)
        x = Flatten()(x)

        top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
        top3_acc.__name__ = 'top3_acc'

        model = Model(inputs=vgg.input, outputs=x)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy', top3_acc])

        return model

    def load_data(self, need_shuffle=True):
        generator = ImageDataGenerator(preprocessing_function=preprocess_input)
        num_samples = len(list(Path(self.data_dir).rglob("*.jpg")))
        self._train_data_generator = generator.flow_from_directory(self.data_dir, target_size=self.image_shape[:2], batch_size=self.batch_size)
        self._train_size = num_samples
        self._valid_data_generator = generator.flow_from_directory(self.data_dir, target_size=self.image_shape[:2], batch_size=self.batch_size)
        self._valid_size = num_samples