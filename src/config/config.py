# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Base Model configurations"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def base_model_config(dataset='PASCAL_VOC'):
  assert dataset.upper()=='PASCAL_VOC' or dataset.upper()=='KITTI', \
      'Currently only support PASCAL_VOC or KITTI dataset'

  cfg = edict()

  # Dataset used to train/val/test model. Now support PASCAL_VOC or KITTI
  cfg.DATASET = dataset.upper()

  if cfg.DATASET == 'PASCAL_VOC':
    # object categories to classify
    cfg.CLASS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                       'sofa', 'train', 'tvmonitor')
  elif cfg.DATASET == 'KITTI':
   # cfg.CLASS_NAMES = ('car', 'pedestrian', 'cyclist')
    # cfg.CLASS_NAMES = ('blue_cup', 'clorox', 'coke', 'detergent', 'downy', 'ranch', 'red_bowl', 'salt', 'scotch_brite', 'spray_bottle', 'sugar', 'sunscreen', 'tide', 'toy', 'waterpot')
    # cfg.CLASS_NAMES = ('006_mustard_bottle', '061_foam_brick', '025_mug',
    #  '021_bleach_cleanser', '051_large_clamp', '035_power_drill', 
    #  '024_bowl', '005_tomato_soup_can', '009_gelatin_box', '004_sugar_box', 
    #  '019_pitcher_base', 'background', '037_scissors', '052_extra_large_clamp',
    #   '040_large_marker', '010_potted_meat_can', '002_master_chef_can', 
    #   '007_tuna_fish_can', '036_wood_block', '008_pudding_box', '003_cracker_box', 
    #   '011_banana')
    cfg.CLASS_NAMES = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', 
      '007_tuna_fish_can','008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
     '019_pitcher_base', '021_bleach_cleanser', '024_bowl','025_mug', 
      '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker','051_large_clamp','052_extra_large_clamp', '061_foam_brick')
  # number of categories to classify
  cfg.CLASSES = len(cfg.CLASS_NAMES)
  print(cfg.CLASSES)    

  # ROI pooling output width
  cfg.GRID_POOL_WIDTH = 7

  # ROI pooling output height
  cfg.GRID_POOL_HEIGHT = 7

  # parameter used in leaky ReLU
  cfg.LEAKY_COEF = 0.1

  # Probability to keep a node in dropout
  cfg.KEEP_PROB = 0.5

  # image width
  cfg.IMAGE_WIDTH = 224

  # image height
  cfg.IMAGE_HEIGHT = 224

  # anchor box, array of [cx, cy, w, h]. To be defined later
  cfg.ANCHOR_BOX = []
  cfg.ANCHOR_BOX2 = []
  cfg.ANCHOR_BOX3 = []

  cfg.ANCHOR_SHAPE       = np.array(
                                [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
                                 [ 162.,  87.], [  38.,  90.], [ 258., 173.],
                                 [ 224., 108.], [  78., 170.], [  72.,  43.]])
  cfg.SCALE_SIZE =1
  # number of anchor boxes
  cfg.ANCHORS = len(cfg.ANCHOR_BOX)
  cfg.ANCHOR_TOTAL = cfg.ANCHORS 

  # number of anchor boxes per grid
  cfg.ANCHOR_PER_GRID = -1

  # batch size
  cfg.BATCH_SIZE = 20

  # Only keep boxes with probability higher than this threshold
  cfg.PROB_THRESH = 0.005

  # Only plot boxes with probability higher than this threshold
  cfg.PLOT_PROB_THRESH = 0.5

  # Bounding boxes with IOU larger than this are going to be removed
  cfg.NMS_THRESH = 0.2

  # Pixel mean values (BGR order) as a (1, 1, 3) array. Below is the BGR mean
  # of VGG16
  cfg.BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])

  # loss coefficient for confidence regression
  cfg.LOSS_COEF_CONF = 1.0

  # loss coefficient for classification regression
  cfg.LOSS_COEF_CLASS = 1.0

  # loss coefficient for bounding box regression
  cfg.LOSS_COEF_BBOX = 10.0
                           
  # reduce step size after this many steps
  cfg.DECAY_STEPS = 10000

  # multiply the learning rate by this factor
  cfg.LR_DECAY_FACTOR = 0.1

  # learning rate
  cfg.LEARNING_RATE = 0.005

  # momentum
  cfg.MOMENTUM = 0.9

  # weight decay
  cfg.WEIGHT_DECAY = 0.0005

  # wether to load pre-trained model
  cfg.LOAD_PRETRAINED_MODEL = True

  # path to load the pre-trained model
  cfg.PRETRAINED_MODEL_PATH = ''

  # print log to console in debug mode
  cfg.DEBUG_MODE = False

  # a small value used to prevent numerical instability
  cfg.EPSILON = 1e-16

  # threshold for safe exponential operation
  cfg.EXP_THRESH=1.0

  # gradients with norm larger than this is going to be clipped.
  cfg.MAX_GRAD_NORM = 10.0

  # Whether to do data augmentation
  cfg.DATA_AUGMENTATION = False

  # The range to randomly shift the image widht
  cfg.DRIFT_X = 0

  # The range to randomly shift the image height
  cfg.DRIFT_Y = 0

  # Whether to exclude images harder than hard-category. Only useful for KITTI
  # dataset.
  cfg.EXCLUDE_HARD_EXAMPLES = True

  # small value used in batch normalization to prevent dividing by 0. The
  # default value here is the same with caffe's default value.
  cfg.BATCH_NORM_EPSILON = 1e-5

  # number of threads to fetch data
  cfg.NUM_THREAD = 4

  # capacity for FIFOQueue
  cfg.QUEUE_CAPACITY = 100

  # indicate if the model is in training mode
  cfg.IS_TRAINING = False

  return cfg
