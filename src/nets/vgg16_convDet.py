# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""VGG16+ConvDet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton_FPN import ModelSkeleton_FPN


class VGG16ConvDet(ModelSkeleton_FPN):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton_FPN.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """Build the VGG-16 model."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      print(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    with tf.variable_scope('conv1') as scope:
      conv1_1 = self._conv_layer(
          'conv1_1', self.image_input, filters=64, size=3, stride=1, freeze=True)
      conv1_2 = self._conv_layer(
          'conv1_2', conv1_1, filters=64, size=3, stride=1, freeze=True)
      pool1 = self._pooling_layer(
          'pool1', conv1_2, size=2, stride=2) #p1

    with tf.variable_scope('conv2') as scope:
      conv2_1 = self._conv_layer(
          'conv2_1', pool1, filters=128, size=3, stride=1, freeze=True)
      conv2_2 = self._conv_layer(
          'conv2_2', conv2_1, filters=128, size=3, stride=1, freeze=True)
      pool2 = self._pooling_layer(
          'pool2', conv2_2, size=2, stride=2) #p2

    with tf.variable_scope('conv3') as scope:
      conv3_1 = self._conv_layer(
          'conv3_1', pool2, filters=256, size=3, stride=1)
      conv3_2 = self._conv_layer(
          'conv3_2', conv3_1, filters=256, size=3, stride=1)
      conv3_3 = self._conv_layer(
          'conv3_3', conv3_2, filters=256, size=3, stride=1)
      pool3 = self._pooling_layer(
          'pool3', conv3_3, size=2, stride=2) #p3

    with tf.variable_scope('conv4') as scope:
      conv4_1 = self._conv_layer(
          'conv4_1', pool3, filters=512, size=3, stride=1)
      conv4_2 = self._conv_layer(
          'conv4_2', conv4_1, filters=512, size=3, stride=1)
      conv4_3 = self._conv_layer(
          'conv4_3', conv4_2, filters=512, size=3, stride=1)
      pool4 = self._pooling_layer(
          'pool4', conv4_3, size=2, stride=2) #p4

    with tf.variable_scope('conv5') as scope:
      conv5_1 = self._conv_layer(
          'conv5_1', pool4, filters=512, size=3, stride=1)
      conv5_2 = self._conv_layer(
          'conv5_2', conv5_1, filters=512, size=3, stride=1)
      conv5_3 = self._conv_layer(
          'conv5_3', conv5_2, filters=512, size=3, stride=1)

    dropout5 = tf.nn.dropout(conv5_3, self.keep_prob, name='drop6')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)


    with tf.variable_scope('feature_pyramid') as scope:
      p5 = self._conv_layer(
        "top_layer", dropout5, filters=256, size=1, stride=1, relu=False)

      lat4 = self._conv_layer('lat4', conv4_1, filters=256, size=1, stride=1, relu=False)
      p4 = self._upsample_add(p5, lat4)
      p4_smooth = self._conv_layer("p4_smooth", p4, filters=256, size=3, stride=1, relu=False)

      lat3 = self._conv_layer("lat3", conv3_1, filters=256, size=1, stride=1, relu=False)
      p3 = self._upsample_add(p4_smooth, lat3)
      p3_smooth = self._conv_layer("p3_smooth", p3 ,filters=256, size=3, stride=1, relu=False)

    # self.feat_pyramid = [p3_smooth, p4_smooth, p5]

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)

    with tf.variable_scope("predictor"):
      # self.preds = tf.layers.conv2d(
      #     p5, filters=num_output, kernel_size=3, padding='SAME', name='pred', reuse=False)
      # self.preds_p4 = tf.layers.conv2d(
      #     p4_smooth, filters=num_output, kernel_size=3, padding='SAME', name='pred', reuse=True)
      # self.preds_p3 = tf.layers.conv2d(
      #     p3_smooth, filters=num_output, kernel_size=3, padding='SAME', name='pred', reuse=True)
    #   self.preds = self._conv_layer(
    #       'pred_p5', p5, filters=num_output, size=3, stride=1,
    #       padding='SAME', xavier=False, relu=False, stddev=0.0001)
    # with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    #   self.preds_p4 = self._conv_layer(
    #         'pred_p4', p4_smooth, filters=num_output, size=3, stride=1,
    #         padding='SAME', xavier=False, relu=False, stddev=0.0001)
    # with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    #   self.preds_p3 = self._conv_layer(
    #         'pred_p3', p3_smooth, filters=num_output, size=3, stride=1,
    #         padding='SAME', xavier=False, relu=False, stddev=0.0001)
      biases = tf.get_variable(
          "bias", [num_output], initializer=tf.constant_initializer(0.0), trainable=True)
      kernel =  tf.get_variable(
          "kernel", [3, 3, 256, num_output], 
          initializer=tf.truncated_normal_initializer(
              stddev=0.0001, dtype=tf.float32),  trainable=True)
      # weight_decay = tf.multiply(tf.nn.l2_loss(kernel), mc.WEIGHT_DECAY, name='weight_loss')
      # tf.add_to_collection('losses', weight_decay)
      self.model_params += [kernel, biases]
      conv = tf.nn.conv2d(
          p5, kernel, [1, 1, 1, 1], padding='SAME',
          name='convolution')
      self.preds = tf.nn.bias_add(conv, biases, name='bias_add')
    with tf.variable_scope("predictor", reuse=True):
      conv = tf.nn.conv2d(
          p4_smooth, kernel, [1, 1, 1, 1], padding='SAME',
          name='convolution')
      self.preds_p4 = tf.nn.bias_add(conv, biases, name='bias_add')
    with tf.variable_scope("predictor", reuse=True):
      conv = tf.nn.conv2d(
          p3_smooth, kernel, [1, 1, 1, 1], padding='SAME',
          name='convolution')
      self.preds_p3 = tf.nn.bias_add(conv, biases, name='bias_add')