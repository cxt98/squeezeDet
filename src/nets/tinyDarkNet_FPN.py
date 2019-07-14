# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet model."""

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
import torch

class tinyDarkNet_FPN(ModelSkeleton_FPN):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton_FPN.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""
    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = torch.load(mc.PRETRAINED_MODEL_PATH)
      for k, v in self.caffemodel_weight.iteritems():
        print (k, v.shape)

   # def _conv_bn_layer(
   #    self, inputs, conv_param_name, bn_param_name, scale_param_name, filters,
   #    size, stride, padding='SAME', freeze=False, relu=True,
   #    conv_with_bias=False, stddev=0.001)

    with tf.variable_scope('conv1') as scope:
        conv1 = self._conv_bn_layer(
           self.image_input, 'conv_0', 'batch_norm_0', filters=16, size=3, stride=1,
          padding='SAME', relu=True)
        pool1 = self._pooling_layer(
          'pool1', conv1, size=2, stride=2) #p1

    with tf.variable_scope('conv2') as scope:
        conv2 = self._conv_bn_layer(
            pool1, 'conv_2', 'batch_norm_2', filters=32 ,size=3, stride=1, padding='SAME', relu=True)
        pool2 = self._pooling_layer(
            'pool2', conv2, size=2, stride=2)

    with tf.variable_scope("conv3") as scope:
        conv3_1 = self._conv_bn_layer(
            pool2, 'conv_4', 'batch_norm_4', filters=16, size=1, stride=1, relu=True)
        conv3_2 = self._conv_bn_layer(
            conv3_1, 'conv_5', "batch_norm_5", filters=128, size=3, stride=1, padding="SAME", relu=True)
        conv3_3 = self._conv_bn_layer(
            conv3_2, 'conv_6', 'batch_norm_6', filters=16, size=1, stride=1, relu=True)
        conv3_4 = self._conv_bn_layer(
            conv3_3, 'conv_7', 'batch_norm_7', filters=128, size=3, stride=1, padding="SAME", relu=True)
        pool3 = self._pooling_layer(
            "pool3", conv3_4, size=2, stride=2)

    with tf.variable_scope("conv4") as scope:
        conv4_1 = self._conv_bn_layer(
            pool3, "conv_9", "batch_norm_9", filters=32, size=1, stride=1, relu=True)
        conv4_2 = self._conv_bn_layer(
            conv4_1, "conv_10", "batch_norm_10", filters=256, size=3, stride=1, padding="SAME", relu=True)
        conv4_3 = self._conv_bn_layer(
            conv4_2, "conv_11", "batch_norm_11", filters=32, size=1, stride=1, relu=True)
        conv4_4 = self._conv_bn_layer(
            conv4_3, "conv_12", "batch_norm_12", filters=256, size=3, stride=1, relu=True)
        pool4 = self._pooling_layer(
            "pool4", conv4_4, size=2, stride=2)

    with tf.variable_scope("conv5") as scope:
        conv5_1 = self._conv_bn_layer(
            pool4, "conv_14", "batch_norm_14", filters=64, size=1, stride=1, relu=True)
        conv5_2 = self._conv_bn_layer(
            conv5_1, "conv_15", "batch_norm_15", filters=512, size=3, stride=1, relu=True)
        conv5_3 = self._conv_bn_layer(
            conv5_2, "conv_16", "batch_norm_16", filters=64, size=1, stride=1, relu=True)
        conv5_4 = self._conv_bn_layer(
            conv5_3, "conv_17", "batch_norm_17", filters=512, size=3, stride=1, relu=True)

    
    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    with tf.variable_scope('feature_pyramid') as scope:
      p5 = self._conv_bn_layer(
        conv5_4, "conv_18", "batch_norm_18", filters=128, size=1, stride=1, relu=False)

      lat4 = self._conv_layer('lat4', conv4_1, filters=128, size=1, stride=1, relu=False)
      p4 = self._upsample_add(p5, lat4)
      p4_smooth = self._conv_layer("p4_smooth", p4, filters=128, size=3, stride=1, relu=False)

      lat3 = self._conv_layer("lat3", conv3_1, filters=128, size=1, stride=1, relu=False)
      p3 = self._upsample_add(p4_smooth, lat3)
      p3_smooth = self._conv_layer("p3_smooth", p3 ,filters=128, size=3, stride=1, relu=False)


    with tf.variable_scope("predictor"):
        biases = tf.get_variable(
          "bias", [num_output], initializer=tf.constant_initializer(0.0), trainable=True)
        kernel =  tf.get_variable(
              "kernel", [3, 3, 128, num_output], 
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