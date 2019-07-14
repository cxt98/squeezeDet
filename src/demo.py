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
      # print(glob.iglob(FLAGS.input_path))
      # for f in glob.iglob(FLAGS.input_path):
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


        det_boxes, det_probs, det_class, class_prob, pred_probs1, pred_probs2, pred_probs3 = sess.run(
            [model.det_boxes, model.det_probs, model.det_class, model.final_class_prob, \
            model.pred_class_probs1, model.pred_class_probs2, model.pred_class_probs3],
            feed_dict={model.image_input:[input_image]})

        # # Filter
        # final_boxes, final_probs, final_class = model.filter_prediction(
        #     det_boxes[0], det_probs[0], det_class[0])
        # keep_idx    = [idx for idx in range(len(final_probs)) \
        #                        if final_probs[idx] > mc.PLOT_PROB_THRESH]
        # final_boxes = [final_boxes[idx] for idx in keep_idx]
        # final_probs = [final_probs[idx] for idx in keep_idx]
        # final_class = [final_class[idx] for idx in keep_idx]
        probs = class_prob[0]
        probs1 = probs[:mc.ANCHORS]
        probs2 = probs[mc.ANCHORS:(mc.ANCHORS+mc.ANCHORS2)]
        probs3 = probs[(mc.ANCHORS+mc.ANCHORS2):]
        probs_all = [probs1, probs2, probs3]
        # heatmap1 = probs1.reshape((30, 40, mc.ANCHOR_PER_GRID, mc.CLASSES))
        # heatmap2 = probs2.reshape((60, 80, mc.ANCHOR_PER_GRID, mc.CLASSES))
        # heatmap3 = probs3.reshape((120,160, mc.ANCHOR_PER_GRID,mc.CLASSES))
        # heatmap1 = probs1.reshape((15, 20, mc.ANCHOR_PER_GRID, mc.CLASSES))
        # heatmap2 = probs2.reshape((30, 40, mc.ANCHOR_PER_GRID, mc.CLASSES))
        # heatmap3 = probs3.reshape((60, 80, mc.ANCHOR_PER_GRID, mc.CLASSES))

        heatmap1 = probs1.reshape((14, 14, mc.ANCHOR_PER_GRID, mc.CLASSES))
        heatmap2 = probs2.reshape((28, 28, mc.ANCHOR_PER_GRID, mc.CLASSES))
        heatmap3 = probs3.reshape((56, 56, mc.ANCHOR_PER_GRID,mc.CLASSES))

        # heatmap1 = probs1.reshape((8, 8, mc.ANCHOR_PER_GRID, mc.CLASSES))
        # heatmap2 = probs2.reshape((16, 16, mc.ANCHOR_PER_GRID, mc.CLASSES))
        # heatmap3 = probs3.reshape((32, 32, mc.ANCHOR_PER_GRID,mc.CLASSES))

        scaled_heatmap = {}
        scaled_heatmap = {
          1: heatmap1,
          2: heatmap2,
          4: heatmap3,
        }
        heatmap_all = [heatmap1, heatmap2, heatmap3]

        bbox_all = det_boxes[0]
        det_box1 = bbox_all[:mc.ANCHORS]
        det_box2 = bbox_all[mc.ANCHORS:(mc.ANCHORS+mc.ANCHORS2)]
        det_box3 = bbox_all[(mc.ANCHORS+mc.ANCHORS2):]


        # print(probs.shape)
        # print(probs1.shape, probs2.shape, probs3.shape)
        # print(probs2[:,:,0,0], probs2[:,:,1,0])

        # print(det_boxes[0].shape)
        # print(det_probs[0].shape)
        # print(det_class[0].shape)
        # print(class_prob[0].shape)

        # bbox_all = det_boxes[0]
        # length = len(det_probs[0])
        # # print(bbox_all[100,:].shape)
        
        # # init heatmap_bbox dict
        heatmap_bbox = {}
        for obj in mc.CLASS_NAMES:
          heatmap_bbox[obj] = np.empty((0,5))
        #create bbox first
        for idx, det_boxes in enumerate([det_box1, det_box2, det_box3]):
          bbox_list = np.empty([0,4])
          for i in range(len(det_boxes)):
            xmin, ymin, xmax, ymax = util.bbox_transform(det_boxes[i,:])
            # print(xmin, ymin, xmax, ymax)
            # print(orig_w/mc.IMAGE_WIDTH)
            xmin = xmin * (float(orig_w) /mc.IMAGE_WIDTH)
            xmax = xmax * (float(orig_w) /mc.IMAGE_WIDTH)
            ymin = ymin * (float(orig_h) /mc.IMAGE_HEIGHT)
            ymax = ymax * (float(orig_h) /mc.IMAGE_HEIGHT) 
            # print(xmin, ymin, xmax, ymax)

            # print(xmin, ymin, xmax, ymax)
            # print(bbox_all[i,:])
            # xmin, ymin, xmax, ymax = bbox_all[i,:]
            bbox_list = np.vstack((bbox_list, np.array([ymin, xmin, ymax, xmax])))


          for cls, cls_name in enumerate(mc.CLASS_NAMES):
            bbox_list_cls = np.hstack((bbox_list, probs_all[idx][:,cls].reshape(-1,1)))
            height, width, _, _ = heatmap_all[idx].shape
            header = np.array([1, width, height,  -1, -1])
            bbox_list_cls = np.vstack((header, bbox_list_cls))
            heatmap_bbox[cls_name] = np.vstack((heatmap_bbox[cls_name], bbox_list_cls))

        bbox_path = os.path.join(FLAGS.out_dir, FLAGS.demo_net+"224",  'bbox')
        heat_path = os.path.join(FLAGS.out_dir, FLAGS.demo_net+"224", "heat")
        if not os.path.exists(bbox_path):
          os.makedirs(bbox_path)
        if not os.path.exists(heat_path):
          os.makedirs(heat_path)


        file_name = scene

        for obj in mc.CLASS_NAMES:  # iter thru each obj
          outpath = bbox_path + '/density_' + \
            file_name + '_' + obj + '.txt'
          np.savetxt(outpath, heatmap_bbox[obj], fmt='%d %d %d %d %.6f')
          print("Save bbox to", outpath)


        
        np.save(heat_path + '/' + file_name + '_heatmap.npy', scaled_heatmap)
        print("Save heat to", heat_path)

        # header = np.array[0, 40, 30, -1, -1]
        # bbox_list = np.vstack(header, bbox_list)

        # for cls, cls_name in enumerate(mc.CLASS_NAMES):

        #   heat_map = class_prob[0][:,cls]
        #   heat_map = heat_1.reshape((30, 40, ANCHOR_PER_GRID))
        #   delta_x, delta_y, delta_w, delta_h = tf.unstack(
        #     self.pred_box_delta, axis=2)
        #   for k in range(ANCHOR_PER_GRID):
        #     heat_1 = heat_map[:,:,k]





        # print(heat_1.shape)
        # ax = sns.heatmap(heat_1[:,:,8])
        # plt.show(ax)
        # plt.figure()

        # TODO(bichen): move this color dict to configuration file
        

        # cls2clr = {
        #     'car': (255, 191, 0),
        #     'cyclist': (0, 191, 255),
        #     'pedestrian':(255, 0, 191)
        # }

        # Draw boxes
        # _draw_box(
        #     im, final_boxes,
        #     [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
        #         for idx, prob in zip(final_class, final_probs)],
        #     # cdict=cls2clr,
        # )

        # file_name = os.path.split(f)[1]
        # out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
        # cv2.imwrite(out_file_name, im)
        # print ('Image detection output saved to {}'.format(out_file_name))


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
  # scene_list = ["exp002_B"]
  image_demo(scene_list, path)

if __name__ == '__main__':
    tf.app.run()
