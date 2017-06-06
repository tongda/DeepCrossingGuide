import csv
import logging
from pathlib import Path
from struct import unpack

import numpy as np
import tensorflow as tf
from keras import activations
from keras.callbacks import Callback, ProgbarLogger, TensorBoard
from keras.layers import (BatchNormalization, Conv2D, Cropping2D, Dense,
                          Dropout, Flatten, Lambda, MaxPooling2D)
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import cv2
from util import read_image, read_metrics

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_dir", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_lpf", True,
                  "Whether to use low pass filter.")
flags.DEFINE_bool("all_feat", True,
                  "Whether to use all features.")
flags.DEFINE_integer("num_epoch", 5,
                     "Number of epochs.")
flags.DEFINE_integer("batch_size", 4,
                     "Number of samples in a batch.")

FLAGS = flags.FLAGS


def feat_size(all_feat=True):
    if all_feat:
        return 12
    else:
        return 3


class CrossingMetrics(object):
    def __init__(self, row, all_feat=True):
        self.track = row[0]
        self.timestamp = row[1]
        self.origin_metrics = row[2:2 + feat_size(all_feat)]
        self.reset_metrics = row[14:14 + feat_size(all_feat)]
        self.filtered_metrics = row[26:26 + feat_size(all_feat)]


class CrossingGuide(object):
    def __init__(self, **conf):
        self.dropout_rate = conf.get("dropout_rate", 0.2)
        self.data_dir = conf.get('data_dir', './data/0524')
        self.activation = activations.get(conf.get('activation', 'relu'))
        self.batch_size = conf.get('batch_size', 128)
        self.use_lpf = conf.get("use_lpf", True)
        self.save_path = conf.get("save_path", "model.h5")
        self.all_feat = conf.get("all_feat", True)

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
        model.add(Flatten())
        model.add(Dense(128, activation=self.activation))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(64, activation=self.activation))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(feat_size(self.all_feat), activation='softmax'))

        model.compile(loss='mse', optimizer='adam')

        return model

    def load_data(self):
        root = Path(self.data_dir)
        with open("./processed.csv", "r") as f:
            reader = csv.reader(f)
            metrics = [CrossingMetrics(row, self.all_feat) for row in reader]

        logging.info("{} sample found.".format(len(metrics)))
        ts_train, ts_valid = train_test_split(metrics, test_size=0.2)

        def generator(samples):
            while True:
                shuffle(samples)
                for offset in range(0, len(samples), self.batch_size):
                    batch_metrics = samples[offset:offset + self.batch_size]
                    images = np.array(
                        [read_image(root / "{}.jpg".format(metric.timestamp)) for metric in batch_metrics])
                    metrics = np.array(
                        [metric.filtered_metrics if self.use_lpf else metric.origin_metrics for metric in batch_metrics])
                    yield images, metrics
        self._train_data_generator = generator(ts_train)
        self._train_size = len(ts_train)
        self._valid_data_generator = generator(ts_valid)
        self._valid_size = len(ts_valid)

    def train(self, num_epoch):
        self.load_data()
        logname = '-'.join(['dr' + str(self.dropout_rate),
                            'ep' + str(num_epoch), activations.serialize(self.activation)])

        tfboard = TensorBoard('./logs/' + logname,
                              histogram_freq=1, write_graph=True)
        self.model.fit_generator(
            self._train_data_generator, steps_per_epoch=self._train_size / self.batch_size,
            validation_data=self._valid_data_generator,
            validation_steps=self._valid_size / self.batch_size,
            epochs=num_epoch, verbose=True,
            max_q_size=50)

        self.model.save(self.save_path)

    def load(self):
        self.model = load_model(self.save_path)

    def predict(self, image):
        return self.model.predict(np.expand_dims(image, 0), batch_size=1)

def main(_):
    print(FLAGS.__dict__['__flags'])
    guide = CrossingGuide(data_path=FLAGS.data_dir, save_path=FLAGS.save_path, use_lpf=FLAGS.use_lpf, batch_size=FLAGS.batch_size, all_feat=FLAGS.all_feat)
    guide.train(FLAGS.num_epoch)

if __name__ == "__main__":
    tf.app.run()
