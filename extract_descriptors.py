"""Evaluate a DeepLab v3 model."""

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

import numpy as np
import timeit

import matplotlib.pyplot as plt

global preds

plt.ioff()

parser = argparse.ArgumentParser()


parser.add_argument('--image_data_dir', type=str, default='../../../../dataset/VOCdevkit/VOC2012/JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--label_data_dir', type=str, default='../../../../dataset/VOCdevkit/VOC2012/SegmentationClassAug',
                    help='The directory containing the ground truth label data.')

parser.add_argument('--evaluation_data_list', type=str, default='../../../../dataset/VOCdevkit/VOC2012/val_20.txt',
                    help='Path to the file listing the evaluation images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")
parser.add_argument('--vgg16_ckpt_path', type=str, default="../../../../Architectures/VGG/vgg_16.ckpt",
                    help='Path to the pre-trained model checkpoint.')
parser.add_argument('--vgg16_npy_path', type=str, default="../../../../Architectures/VGG/vgg16.npy",
                    help='Path to the npy pre-trained model.')
#----------------------------------------------------------------------------#
parser.add_argument('--keep_prob', type=float, default=1.0, # no matter it will be bypass
                    help='Dropout keep probability.')
#----------------------------------------------------------------------------#
parser.add_argument('--weight_decay', type=float, default=2e-4, 
                    help='The weight decay to use for regularizing the model.')

parser.add_argument('--extracted_descriptors_dir', type=str, default='./extracted_descriptors',
                    help='Path to the extracted descriptors.')

_NUM_CLASSES = 21

class_name = ['background',
                 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                 'bus', 'car', 'cat', 'chair', 'cow',
                 'dining table', 'dog', 'horse', 'motorbike', 'person',
                 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def main(unused_argv):
  
  global preds, features_val, labels_val
  global net_concat, lambdas, image_level_features_256
  global b1u3conv1, b1u3conv3, b2u4conv3, b3u6conv3, b4u3conv3
  global low_level_features, decoded_labels

  if not os.path.exists(FLAGS.extracted_descriptors_dir):
    os.makedirs(FLAGS.extracted_descriptors_dir)
  
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  examples = dataset_util.read_examples_list(FLAGS.evaluation_data_list)
  image_files = [os.path.join(FLAGS.image_data_dir, filename) + '.jpg' for filename in examples]
  label_files = [os.path.join(FLAGS.label_data_dir, filename) + '.png' for filename in examples]

  features, labels = preprocessing.eval_input_fn(image_files, label_files)
  #print(features, labels)

  predictions = fcn8_vgg_model.model_fn(
      features,
      labels,
      tf.estimator.ModeKeys.EVAL,
      params={
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'num_classes': _NUM_CLASSES,
          'weight_decay': FLAGS.weight_decay,
          'vgg16_ckpt_path': FLAGS.vgg16_ckpt_path,
          'vgg16_npy_path': FLAGS.vgg16_npy_path,
          'keep_prob': FLAGS.keep_prob,
      }).predictions
  
  #print('predictions', predictions)

  # Manually load the latest checkpoint
  saver = tf.train.Saver()
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    print("\n***** Aggregating descriptors *****")
    print("Dataset --> VOC2012")
    print("Model --> FCN8-VGG16")
    print("")

    descriptors = []
    start = timeit.default_timer()
    n_saving = 26 # 10582 = 13*2*407
    itr = 0
    # Loop through the batches and store predictions and labels
    while True:
      try:
        itr +=1
        preds, features_val, labels_val = sess.run([predictions, features, labels])
        print(itr, preds['descriptors'].shape)
        descriptors.append(preds['descriptors'][0])
        if ((itr % n_saving) == 0):
          descriptors_name = 'descriptors_' + str(itr//n_saving) + '.npz'
          descriptors_path = os.path.join(FLAGS.extracted_descriptors_dir,descriptors_name)
          np.savez(descriptors_path, descriptors = descriptors)
          stop = timeit.default_timer()
          print("Step = {} ({:.3f} sec)".format(itr//n_saving, stop-start))
          start = timeit.default_timer()
          #print('--------------------------------------------------------')
      
          descriptors = []
          
      except tf.errors.OutOfRangeError:
        break



  
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
