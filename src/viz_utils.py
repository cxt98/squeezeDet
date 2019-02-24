# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *
from PIL import Image
import matplotlib.pyplot as plt
from utils import util


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")
tf.app.flags.DEFINE_string(
    'test_cls', 'blue_cup', """class for testing""")

class_labels =  {'blue_cup' : 0,
 'clorox': 1, 'coke': 2, 'detergent': 3, 'downy': 4, 'ranch': 5, 'red_bowl': 6, 'salt': 7,
  'scotch_brite':8, 'spray_bottle': 9, 'sugar': 10, 'sunscreen': 11, 'tide': 12, 'toy': 13, 'waterpot': 14}

def display_heatmap(mc):
  image_path = os.path.join(FLAGS.input_path)
  img = Image.open(image_path)
  file_name = os.path.split(FLAGS.input_path)[1]
  file_name = file_name.split('.')[0]  # load heatmap
  heatmap = np.load(os.path.join(FLAGS.out_dir, "heatmap") + '/' + file_name+'_heatmap.npy')
  plt.figure()
  plt.axis('off')
  plt.imshow(np.asarray(img))
  plt.figure()
  for k in range(mc.ANCHOR_PER_GRID):
    print('shape: {}'.format(k))
    cls_out_np = heatmap[:,:,k,:]
    scale_map = cls_out_np[:, :, class_labels[FLAGS.test_cls]]
    plt.subplot(3, 3, k+1)
    plt.title(FLAGS.test_cls)
    plt.imshow(scale_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
  plt.show()


def image_demo():
  """Detect image."""

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)
      print(FLAGS.input_path)
      print(glob.iglob(FLAGS.input_path))
      for f in glob.iglob(FLAGS.input_path):
        im = cv2.imread(f)
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        input_image = im - mc.BGR_MEANS

        # Detect
        det_boxes, det_probs, det_class, class_prob = sess.run(
            [model.det_boxes, model.det_probs, model.det_class, model.final_class_prob],
            feed_dict={model.image_input:[input_image]})

        # Filter
        if FLAGS.mode == "bbox":
          final_boxes, final_probs, final_class = model.filter_prediction(
              det_boxes[0], det_probs[0], det_class[0])
          keep_idx    = [idx for idx in range(len(final_probs)) \
                            if final_probs[idx] > mc.PLOT_PROB_THRESH]
          final_boxes = [final_boxes[idx] for idx in keep_idx]
          final_probs = [final_probs[idx] for idx in keep_idx]
          final_class = [final_class[idx] for idx in keep_idx]

          # Draw boxes
          _draw_box(
              im, final_boxes,
              [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                  for idx, prob in zip(final_class, final_probs)],
          )

          file_name = os.path.split(f)[1]
          out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
          cv2.imwrite(out_file_name, im)
          print ('Image detection output saved to {}'.format(out_file_name))

        if FLAGS.mode == "heatmap":
          display_heatmap(mc)



def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  image_demo()

if __name__ == '__main__':
    tf.app.run()
