"""Train a DeepLab v3 plus model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from utils import preprocessing
from tensorflow.python import debug as tf_debug

import shutil
import platform

#sys.path.append("models")
#from xception_model import build_XNET
#import fcn32_vgg as fcn_vgg
import fcn8_vgg_model
#import fcn32_vgg


_NUM_CLASSES = 21
_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4
_POWER = 0.9
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}

_MAX_ITER = 120000

uname = platform.uname()[1]
if uname == 'family-pc':
  _BATCH_SIZE = 1
  _CROP_HIGHT = 33
  _CROP_WIDTH = 33
elif uname == 'deep-pc':
  _BATCH_SIZE = 9
  _CROP_HIGHT = 385
  _CROP_WIDTH = 385
else:
  _BATCH_SIZE = 15
  _CROP_HIGHT = 448
  _CROP_WIDTH = 448

_NUM_EPOCHS = (_MAX_ITER * _BATCH_SIZE // _NUM_IMAGES['train']) + 1

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument('--Netvlad_K', type=int, default=32,
                    help='Number of centers in Netvlad layer.')
parser.add_argument('--Netvlad_centers', type=str, default='./kmeans_centers/centers_32x21.npz',
                    help='directory for Netvlad centers.')
#parser.add_argument('--Netvlad_centers', type=str, default=None,
#                    help='directory for Netvlad centers.')
parser.add_argument('--is_CAF', type=bool, default=True,
                    help='')

parser.add_argument('--train_epochs', type=int, default=_NUM_EPOCHS,
                    help='Number of training epochs: '
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                         'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                         'For 30K iteration with batch size 9, train_epoch = 68.04  (= 80K * 9 / 10,582). '
                         'For 45K iteration with batch size 9, train_epoch = 38.27  (= 45K * 9 / 10,582). '
                         'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                         'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                         'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                         'For 45K iteration with batch size 15, train_epoch = 63.78 (= 45K * 15 / 10,582).'
                         'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582). '
                         )

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')
parser.add_argument('--crop_height', type=int, default=_CROP_HIGHT, 
                    help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=_CROP_WIDTH, 
                    help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=_BATCH_SIZE, 
                    help='Number of images in each batch')
parser.add_argument('--max_iter', type=int, default=_MAX_ITER, 
                    help='Number of maximum iteration used for "poly" learning rate policy.')


parser.add_argument('--keep_prob', type=float, default=0.5, 
                    help='Dropout keep probability.')
parser.add_argument('--learning_rate_policy', type=str, default='constant',
                    choices=['poly', 'piecewise', 'constant'],
                    help='Learning rate policy to optimize loss.')
parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')
parser.add_argument('--model_dir', type=str, default='./model',
                    help='Base directory for the model.')
parser.add_argument('--vgg16_ckpt_path', type=str, default="../../../../Architectures/VGG/vgg_16.ckpt",
                    help='Path to the pre-trained model checkpoint.')
parser.add_argument('--vgg16_npy_path', type=str, default="../../../../Architectures/VGG/vgg16.npy",
                    help='Path to the npy pre-trained model.')
parser.add_argument('--data_dir', type=str, default='../../../../dataset/VOCdevkit/VOC2012/',
                    help='Path to the directory containing the PASCAL VOC data tf record.')
parser.add_argument('--initial_learning_rate', type=float, default=1e-4, #7e-3, 
                    help='Initial learning rate for the optimizer.')
parser.add_argument('--end_learning_rate', type=float, default=1e-5, #1e-6, 
                    help='End learning rate for the optimizer.')
parser.add_argument('--initial_global_step', type=int, default=0, 
                    help='Initial global step for controlling learning rate when fine-tuning model.')
parser.add_argument('--weight_decay', type=float, default=2e-4, 
                    help='The weight decay to use for regularizing the model.')
parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')


#_HEIGHT = 513   #513
#_WIDTH = 513    #513
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255


_BATCH_NORM_DECAY = 0.9997


def get_filenames(is_training, data_dir):
  """Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  if is_training:
    return [os.path.join(data_dir, 'voc_train.record')]
  else:
    return [os.path.join(data_dir, 'voc_val.record')]


def parse_record(raw_record):
  """Parse PASCAL image and label from a tf record."""
  keys_to_features = {
      'image/height':
      tf.FixedLenFeature((), tf.int64),
      'image/width':
      tf.FixedLenFeature((), tf.int64),
      'image/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
      tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'label/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
  }

  parsed = tf.parse_single_example(raw_record, keys_to_features)

  # height = tf.cast(parsed['image/height'], tf.int32)
  # width = tf.cast(parsed['image/width'], tf.int32)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]), _DEPTH)
  image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
  image.set_shape([None, None, 3])

  label = tf.image.decode_image(
      tf.reshape(parsed['label/encoded'], shape=[]), 1)
  label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
  label.set_shape([None, None, 1])

  return image, label


def preprocess_image(image, label, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Randomly scale the image and label.
    image, label = preprocessing.random_rescale_image_and_label(
        image, label, _MIN_SCALE, _MAX_SCALE)

    # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
    image, label = preprocessing.random_crop_or_pad_image_and_label(
        image, label, _CROP_HIGHT, _CROP_WIDTH, _IGNORE_LABEL)

    # Randomly flip the image and label horizontally.
    image, label = preprocessing.random_flip_left_right_image_and_label(
        image, label)

    image.set_shape([_CROP_HIGHT, _CROP_WIDTH, 3])
    label.set_shape([_CROP_HIGHT, _CROP_WIDTH, 1])

  image = preprocessing.mean_image_subtraction(image)

  return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  
  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: preprocess_image(image, label, is_training))
  dataset = dataset.prefetch(batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if FLAGS.clean_model_dir:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig(save_checkpoints_secs=1e9,
                                      keep_checkpoint_max=5,).replace()

  
  model = tf.estimator.Estimator(
      model_fn=fcn8_vgg_model.model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config,
      params={
          'batch_size': FLAGS.batch_size,
          'num_classes': _NUM_CLASSES,
          'weight_decay': FLAGS.weight_decay,
          'learning_rate_policy': FLAGS.learning_rate_policy,
          'num_train': _NUM_IMAGES['train'],
          'initial_learning_rate': FLAGS.initial_learning_rate,
          'max_iter': FLAGS.max_iter,
          'end_learning_rate': FLAGS.end_learning_rate,
          'power': _POWER,
          'momentum': _MOMENTUM,
          'initial_global_step': FLAGS.initial_global_step,
          'vgg16_ckpt_path': FLAGS.vgg16_ckpt_path,
          'vgg16_npy_path': FLAGS.vgg16_npy_path,
          'keep_prob': FLAGS.keep_prob,
          'is_CAF': FLAGS.is_CAF,
          'Netvlad_K': FLAGS.Netvlad_K,
          'Netvlad_centers': FLAGS.Netvlad_centers,
      })

  if FLAGS.is_CAF is True:
    architecture = 'FCN8_CAF'
  else:
    architecture = 'FCN8'
    
  print('Architecture is : ', architecture)
  n_tstep = FLAGS.train_epochs // FLAGS.epochs_per_eval
  for tstep in range(n_tstep):
    tf.logging.info("-------------------------------------------------------------------------")
    tf.logging.info('train step = {0}/{1}'.format(tstep, n_tstep))
    tf.logging.info("-------------------------------------------------------------------------")

    tensors_to_log = {
      'learning_rate': 'learning_rate',
      'cross_entropy': 'cross_entropy',
      'train_px_accuracy': 'train_px_accuracy',
      'train_mean_iou': 'train_mean_iou',
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    train_hooks = [logging_hook]
    eval_hooks = None

    if FLAGS.debug:
      debug_hook = tf_debug.LocalCLIDebugHook()
      train_hooks.append(debug_hook)
      eval_hooks = [debug_hook]


    tf.logging.info("Start training.")
    model.train(
        input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=train_hooks,
        #steps=1  # For debug
        )


    tf.logging.info("Start evaluation.")
    # Evaluate the model and print results
    eval_results = model.evaluate(
        # Batch size must be 1 for testing because the images' size differs
        input_fn=lambda: input_fn(False, FLAGS.data_dir, 1),
        hooks=eval_hooks,
        #steps=1  # For debug
        )
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
