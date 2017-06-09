import tensorflow as tf
from crossing_guide import CrossingGuide

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_dir", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string("piece_file", None,
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


def main(_):
    print(FLAGS.__dict__['__flags'])
    guide = CrossingGuide(data_dir=FLAGS.data_dir,
                          save_path=FLAGS.save_path,
                          use_lpf=FLAGS.use_lpf,
                          batch_size=FLAGS.batch_size,
                          all_feat=FLAGS.all_feat,
                          piece_file=FLAGS.piece_file,
                          activation='tanh')
    guide.train(FLAGS.num_epoch)


if __name__ == "__main__":
    tf.app.run()
