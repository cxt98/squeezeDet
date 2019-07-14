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
tf.app.flags.DEFINE_float(
    'cls_thresh', 0.6, """class prob threshold""")
tf.app.flags.DEFINE_string(
    'scene', 'exp001_B', """scene name""")

# class_labels =  {'blue_cup' : 0,
#  'clorox': 1, 'coke': 2, 'detergent': 3, 'downy': 4, 'ranch': 5, 'red_bowl': 6, 'salt': 7,
#   'scotch_brite':8, 'spray_bottle': 9, 'sugar': 10, 'sunscreen': 11, 'tide': 12, 'toy': 13, 'waterpot': 14}
class_labels = {'006_mustard_bottle': 4, '061_foam_brick': 20, '025_mug': 13,
     '021_bleach_cleanser': 11, '051_large_clamp': 18, '035_power_drill': 14, 
     '024_bowl': 12, '005_tomato_soup_can': 3, '009_gelatin_box': 7, '004_sugar_box': 2, 
     '019_pitcher_base': 10, 'background': 21, '037_scissors': 16, '052_extra_large_clamp': 19,
      '040_large_marker': 17, '010_potted_meat_can': 8, '002_master_chef_can': 0, 
      '007_tuna_fish_can': 5, '036_wood_block': 15, '008_pudding_box': 6, '003_cracker_box': 1, 
      '011_banana': 9}

def read_gt(scene, opts):
    """ Read ground truth from txt file """
    result_path = os.path.join(
        opts.gt_path, scene, 'gt_pose_zyx.txt')
    gt_bbox = pd.read_csv(result_path, sep=' ', header=None)
    gt_bbox = np.array(gt_bbox)
    if gt_bbox.ndim == 1:
        gt_bbox = np.array([gt_bbox])
    obj_idx = np.where(gt_bbox[:, 0] == opts.test_cls)

    gt_bbox = gt_bbox[obj_idx[0], -4:]  # keep only the bboxes
    # convert the bbox from [xmin xmax ymin ymax] to [ymin xmin ymax xmax]
    gt_bbox = np.hstack((gt_bbox[:, 2].reshape(-1, 1), gt_bbox[:, 0].reshape(-1, 1),
                         gt_bbox[:, 3].reshape(-1, 1), gt_bbox[:, 1].reshape(-1, 1)))

    gt_bbox = np.hstack((gt_bbox, np.ones((gt_bbox.shape[0], 1))))
    return gt_bbox, img_path

def draw_box(im, box_list, label_list, color=(0,255,0), cdict=None, form='center'):
  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  for bbox, label in zip(box_list, label_list):

    if form == 'center':
      bbox = bbox_transform(bbox)

    # xmin, ymin, xmax, ymax = [int(b) for b in bbox]
    xmin, xmax, ymin, ymax  = [int(b) for b in bbox]

    l = label.split(':')[0] # text before "CLASS: (PROB)"
    if cdict and l in cdict:
      c = cdict[l]
    else:
      c = color

    # draw box
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
    # draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)



def viz_detections(img, class_name, dets, thresh=0.5, b_color='red', ax=None, u_thresh=1.0):
    """Draw detected bounding boxes."""
    if dets.ndim == 1:
        dets = np.array([dets])
    inds = np.where(dets[:, -1] >= thresh)[0]
    u_idx = np.where(dets[:, -1] <= u_thresh)[0]
    inds = np.intersect1d(inds, u_idx)
    if len(inds) == 0:  # check empty inds
        return

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(np.asarray(img), aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        # bbox[0] = bbox[0]*(480./224)
        # bbox[1] = bbox[1]*(640./224)
        # bbox[2] = bbox[2]*(480./224)
        # bbox[3] = bbox[3]*(640./224)

        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[1], bbox[0]),  # xy
                          bbox[3] - bbox[1],  # width
                          bbox[2] - bbox[0],  # height
                          fill=False,
                          edgecolor=b_color, linewidth=3.5)
        )
        ax.text(bbox[1] - 2, bbox[0],
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.4f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def filter_prediction(mc, boxes, probs, cls_idx):
    if mc.TOP_N_DETECTION < len(probs) and mc.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-mc.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
    else:
      filtered_idx = np.nonzero(probs>mc.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]

    final_boxes = []
    final_probs = []
    final_cls_idx = []

    for c in range(mc.CLASSES):
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
      keep = util.nms(boxes[idx_per_class], probs[idx_per_class], mc.NMS_THRESH)
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)
    return final_boxes, final_probs, final_cls_idx

def display_heatmap(mc):
  image_path = os.path.join('/home/liuyanqi/caffe/pyramid_cnn/data/adversarial/', FLAGS.scene, 'scenergb.jpg')
  img = Image.open(image_path)
  # file_name = os.path.split(FLAGS.input_path)[1]
  # file_name = file_name.split('.')[0]  # load heatmap
  file_name = FLAGS.scene
  scaled_heatmap = np.load(os.path.join(FLAGS.out_dir) + '/' + file_name+'_heatmap.npy')
  plt.figure()
  plt.subplot2grid((3, 3), (0,0), colspan=2, rowspan=2)
  plt.axis('off')
  plt.imshow(np.asarray(img))
  for idx, s in enumerate([1, 2, 4]):
    heatmap = scaled_heatmap[()][s]
    print(heatmap.shape)
    for k in range(mc.ANCHOR_PER_GRID):
      print('shape: {}'.format(k))
      cls_out_np = heatmap[:,:,k,:]
      scale_map = cls_out_np[:, :, class_labels[FLAGS.test_cls]]
      # plt.figure()
      plt.subplot2grid((3, 3), (2,idx))
      plt.title(FLAGS.test_cls)
      plt.imshow(scale_map, cmap='hot', interpolation='nearest')
      plt.colorbar()

  plt.show()


def display_bbox(mc):

    gt_path = "/home/liuyanqi/caffe/pyramid_cnn/data/adversarial/"
    # img_path = os.path.join(gt_path, scene, 'scenergb.jpg')
    img_path = gt_path + '/' + FLAGS.scene + '/scenergb.jpg';
    im = cv2.imread(img_path)
    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
    # input_image = im - mc.BGR_MEANS
    # read bbox file
    det_boxes = np.empty([0,4])
    det_probs = np.empty([0])
    det_class = np.full((1,1), 0)
    # for test_id, test_cls in enumerate(mc.CLASS_NAMES):
    dets = np.genfromtxt(os.path.join(FLAGS.out_dir) + '/' +
                         'density' + '_' + FLAGS.scene +
                         '_' + FLAGS.test_cls + '.txt')
    # remove the trash
    trash_idx = np.nonzero(dets < 0)[0][::2]
    dets = np.delete(dets, trash_idx, axis=0)


    # use nms to prone dets
    # keep = selective_nms(dets, opts)

    nms_dets = dets

    # viz ground truth
    img = Image.open(img_path)
    # img = img.resize((mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
    _, ax = plt.subplots(figsize=(12, 12))
    # viz_detections(img, FLAGS.test_cls, gt_bbox,
    #                thresh=FLAGS.cls_thresh, b_color='red', ax=ax)
    # viz_det_paper(img, opts.test_cls, gt_bbox,
    #                thresh=opts.cls_thresh, b_color='red', ax=ax)


    viz_detections(img, FLAGS.test_cls, dets,
                   thresh=FLAGS.cls_thresh, b_color='blue', ax=ax, u_thresh=1.0)
    # viz_det_paper(img, opts.test_cls, nms_dets,
    #                thresh=opts.cls_thresh, b_color='blue', ax=ax, u_thresh=opts.u_thresh)

    plt.show()



    # print(dets.shape)
    # det_class = np.vstack((det_class, np.full((10800,1), test_id)))

    # det_boxes = np.vstack((det_boxes, dets[:, :-1]))
    # det_probs = np.concatenate((det_probs, dets[:, -1]))

    # print(det_class.shape, det_boxes.shape, det_probs.shape);

    # final_boxes, final_probs, final_class = det_boxes, det_probs, det_class[1:,0]
    # final_boxes, final_probs, final_class = filter_prediction(mc, det_boxes, det_probs, det_class[1:])
    # keep_idx    = [idx for idx in range(len(final_probs)) \
    #                   if final_probs[idx] > opts.cls_thresh]
    # final_boxes = [final_boxes[idx] for idx in keep_idx]
    # final_probs = [final_probs[idx] for idx in keep_idx]
    # final_class = [final_class[idx] for idx in keep_idx]
    
    # final_boxes = np.array(final_boxes)
    # new_final_boxes = []
    # # for i in range(len(final_probs)):
    # #   new_final_boxes.append(util.bbox_transform_inv(final_boxes[i,:]))
    
    # Draw boxes
    # _draw_box(
    #     im, final_boxes,
    #     [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
    #         for idx, prob in zip(final_class, final_probs)],
    # )

    # file_name = FLAGS.scene+'.jpg'
    # out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
    # cv2.imwrite(out_file_name, im)
    # # plt.imshow(im)
    # # plt.show()
    # print ('Image detection output saved to {}'.format(out_file_name))






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
          # final_boxes, final_probs, final_class = model.filter_prediction(
          #     det_boxes[0], det_probs[0], det_class[0])
          keep_idx    = [idx for idx in range(len(final_probs)) \
                            if final_probs[idx] > mc.PLOT_PROB_THRESH]
          final_boxes = [final_boxes[idx] for idx in keep_idx]
          final_probs = [final_probs[idx] for idx in keep_idx]
          final_class = [final_class[idx] for idx in keep_idx]
          
          print(final_class, final_probs, np.array(final_boxes));

          # Draw boxes
          draw_box(
              im, final_boxes,
              [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                  for idx, prob in zip(final_class, final_probs)],
          )

          file_name = os.path.split(f)[1]
          out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
          cv2.imwrite(out_file_name, im)
          print ('Image detection output saved to {}'.format(out_file_name))





def main(argv=None):

  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  mc = kitti_vgg16_config()
  mc.LOAD_PRETRAINED_MODEL = False

  if FLAGS.mode == "heatmap":
    display_heatmap(mc)
  else:
    # image_demo()
    display_bbox(mc)

if __name__ == '__main__':
    tf.app.run()
