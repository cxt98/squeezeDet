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
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from utils import util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', '/tmp/logs/+zynqDet/train/model_11999.ckpt-11999',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'zynqDet', """Neural net architecture.""")


def image_demo(scene_list, path):
  """Detect image."""

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
    elif FLAGS.demo_net == 'zynqDet':
      mc = kitti_zynqDet_FPN_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = ZynqDet_FPN(mc, FLAGS.gpu)
    elif FLAGS.demo_net == "vgg16":
      mc = kitti_vgg16_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = VGG16ConvDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == "yolo":
      mc = kitti_vgg16_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = tinyDarkNet_FPN(mc, FLAGS.gpu)
    else:
      print("no such model")

    saver = tf.train.Saver(model.model_params)
    print(FLAGS.checkpoint)
    print(FLAGS.out_dir)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)
      print(path)

      tensor = tf.get_default_graph().get_tensor_by_name('conv1/conv1/kernels:0')
      print(sess.run(tensor[:,:,0,0]))
      
      for scene in scene_list:
        image_path = os.path.join(path, scene, 'scenergb.jpg')
        im = cv2.imread(image_path)
        im = im.astype(np.float32, copy=False)
        orig_h, orig_w, _ = [float(v) for v in im.shape]

        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        input_image = im - mc.BGR_MEANS

        # Detect
        x_scale = mc.IMAGE_WIDTH/orig_w
        y_scale = mc.IMAGE_HEIGHT/orig_h

        # det_boxes, det_probs, det_class, class_prob, pred_probs1, pred_probs2, pred_probs3 = sess.run(
        #     [model.det_boxes, model.det_probs, model.det_class, model.final_class_prob, \
        #     model.pred_class_probs1, model.pred_class_probs2, model.pred_class_probs3],
        #     feed_dict={model.image_input:[input_image]})
        print(scene)
        print(input_image[0,0,:])
        conv1, preds, preds_p4, preds_p3, \
        det_boxes, class_prob, pred_probs1,\
        pred_conf1, pred_bbox_delta1 = sess.run([model.conv1, model.preds, model.preds_p4, \
                                                                            model.preds_p3, model.det_boxes, model.final_class_prob,\
                                                                            model.pred_class_probs1, model.pred_conf1, model.pred_bbox_delta1], 
                                                                            feed_dict={model.image_input:[input_image]})

        print(conv1.shape)
        print(conv1[0,1,1,:])
        print(preds.shape)
        print(preds[0,:,:,22:])
        # print(preds_p4.shape)
        # print(preds_p4[0,1,0, :])
        # print(preds_p3.shape)
        # print(preds_p3[0,1,0,:])

        bbox_all = det_boxes[0]
        det_box1 = bbox_all[:mc.ANCHORS]
        det_box2 = bbox_all[mc.ANCHORS:(mc.ANCHORS+mc.ANCHORS2)]
        det_box3 = bbox_all[(mc.ANCHORS+mc.ANCHORS2):]

        # print(det_box1.shape)
        # print(det_box1[0,:])

        probs = class_prob[0]
        probs1 = probs[:mc.ANCHORS]
        probs2 = probs[mc.ANCHORS:(mc.ANCHORS+mc.ANCHORS2)]
        probs3 = probs[(mc.ANCHORS+mc.ANCHORS2):]

        print(pred_probs1.shape)
        print(pred_probs1[0,0,:])
        print(np.sum(pred_probs1[0,0,:]))

        print(pred_conf1.shape)
        print(pred_conf1[0,:])

        print(pred_bbox_delta1.shape)
        print(pred_bbox_delta1[0,0,:])

        # print(preds[:,:,:,22:])

        print(probs1.shape)
        print(probs1[0,:])

        print(det_box1.shape)
        print(det_box1[0,:])

def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  # scene_variation = ["B"]
  scene_variation = ["B", "D", "D1", "D2", "O1", "O2", "O3"]
  scene_list = []
  test_set = list(range(1, 41))
  for v in scene_variation:
      for i in test_set:
          scene_list.append('exp{:03d}_{}'.format(i, v))
  path = '/home/liuyanqi/caffe/pyramid_cnn/data/adversarial'
  scene_list = ["exp001_B"]
  image_demo(scene_list, path)

if __name__ == '__main__':
    tf.app.run()
