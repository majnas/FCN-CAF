"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import fcn8_vgg_model
from utils import preprocessing
from utils import dataset_util

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug

_MAX_ITER = 120000
_NUM_CLASSES = 21
_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4
_POWER = 0.9
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='../../../../dataset/VOCdevkit/VOC2012/JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='./inference_output',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='../../../../dataset/VOCdevkit/VOC2012/inf_100.txt',
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

#parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
#                    choices=['resnet_v2_50', 'resnet_v2_101'],
#                    help='The architecture of base Resnet building block.')

#parser.add_argument('--output_stride', type=int, default=16,
#                    choices=[8, 16],
#                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--max_iter', type=int, default=_MAX_ITER, 
                    help='Number of maximum iteration used for "poly" learning rate policy.')
parser.add_argument('--keep_prob', type=float, default=0.5, 
                    help='Dropout keep probability.')
parser.add_argument('--learning_rate_policy', type=str, default='constant',
                    choices=['poly', 'piecewise', 'constant'],
                    help='Learning rate policy to optimize loss.')
parser.add_argument('--vgg16_ckpt_path', type=str, default="../../../../Architectures/VGG/vgg_16.ckpt",
                    help='Path to the pre-trained model checkpoint.')
parser.add_argument('--vgg16_npy_path', type=str, default="../../../../Architectures/VGG/vgg16.npy",
                    help='Path to the npy pre-trained model.')
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


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=fcn8_vgg_model.model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
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
      })

  examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
  image_files = [os.path.join(FLAGS.data_dir, filename) for filename in examples]
  print(image_files)
  
  predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files),
        hooks=pred_hooks)

  output_dir = FLAGS.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for pred_dict, image_path in zip(predictions, image_files):
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = image_basename + '_mask.png'
    path_to_output = os.path.join(output_dir, output_filename)

    print("generating:", path_to_output)
    mask = pred_dict['decoded_labels']
    print('mask', mask.shape)
    mask = Image.fromarray(mask)
    mask.save(path_to_output)
    #plt.axis('off')
    #plt.imshow(mask)
    #plt.savefig(path_to_output, bbox_inches='tight')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
