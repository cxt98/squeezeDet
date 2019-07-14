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

class ZynqDet_FPN(ModelSkeleton_FPN):
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
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)
      for k, v in self.caffemodel_weight.iteritems():
        print (k, v.shape)
    with tf.variable_scope('conv1')as scope:
      self.conv1 = self._conv_layer(
          'conv1', self.image_input, filters=64, size=3, stride=2,
          padding='SAME', freeze=True, relu=False)

    with tf.variable_scope("conv2") as scope:
      fire2 = self._fire_layer(
          'fire2', self.conv1, s1x1=16, e1x1=64, e3x3=64, pool=True, freeze=False)
      fire3 = self._fire_layer(
          'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)

    with tf.variable_scope("conv3") as scope:
      fire4 = self._fire_layer(
          'fire4', fire3, s1x1=32, e1x1=128, e3x3=128, pool=True, freeze=False)
      fire5 = self._fire_layer(
          'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)

    with tf.variable_scope("conv4") as scope:
      fire6 = self._fire_layer(
          'fire6', fire5, s1x1=64, e1x1=256, e3x3=256, pool=True, freeze=False)
      fire7 = self._fire_layer(
          'fire7', fire6, s1x1=64, e1x1=192, e3x3=192, freeze=False)
    
    with tf.variable_scope("conv5") as scope:
      fire8 = self._fire_layer(
          'fire8', fire7, s1x1=112, e1x1=256, e3x3=256, pool=True, freeze=False)
      fire9 = self._fire_layer(
          'fire9', fire8, s1x1=112, e1x1=368, e3x3=368, freeze=False)

      # # # Two extra fire modules that are not trained before
      # fire10 = self._fire_layer(
      #     'fire10', fire9, s1x1=112, e1x1=368, e3x3=368, freeze=False)
      # fire11 = self._fire_layer(
      #     'fire11', fire10, s1x1=112, e1x1=368, e3x3=368, freeze=False)
      # dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')


    with tf.variable_scope("feature_pyramid") as scope:
      p5 = self._conv_layer(
        "top_layer", fire9, filters=256, size=1, stride=1, relu=False)

      lat4 = self._conv_layer('lat4', fire6, filters=256, size=1, stride=1, relu=False) #operation: 30*40*256*512
      p4 = self._upsample_add(p5, lat4) #operation 256*30*40
      # p4_smooth = self._conv_layer("p4_smooth", p4, filters=256, size=3, stride=1, relu=False) #operation: 30*40*256*256*9

      lat3 = self._conv_layer("lat3", fire4, filters=256, size=1, stride=1, relu=False) # 60*80*256*256
      p3 = self._upsample_add(p4, lat3) #operation 256*60*80
      # p3_smooth = self._conv_layer("p3_smooth", p3 ,filters=256, size=3, stride=1, relu=False) #60*80*256*256*9
    

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    with tf.variable_scope("predictor") as scope: #with tf.variable_scope("prediction", reuse=True):
      # self.preds = self._conv_layer(
      #   'pred_p5', p5, filters=num_output, size=3, stride=1,
      #   padding='SAME', xavier=False, relu=False, stddev=0.0001)
      # self.preds_p4 = self._conv_layer(
      #     'pred_p4', p4_smooth, filters=num_output, size=3, stride=1,
      #     padding='SAME', xavier=False, relu=False, stddev=0.0001)
      # self.preds_p3 = self._conv_layer(
      #     'pred_p3', p3_smooth, filters=num_output, size=3, stride=1,
      #     padding='SAME', xavier=False, relu=False, stddev=0.0001)
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
      self.preds = tf.nn.bias_add(conv, biases, name='bias_add') #15*20*256*26*9
    with tf.variable_scope("predictor", reuse=True):
      conv = tf.nn.conv2d(
          p4, kernel, [1, 1, 1, 1], padding='SAME',
          name='convolution')
      self.preds_p4 = tf.nn.bias_add(conv, biases, name='bias_add') #30*40*256*26*9
    with tf.variable_scope("predictor", reuse=True):
      conv = tf.nn.conv2d(
          p3, kernel, [1, 1, 1, 1], padding='SAME',
          name='convolution')
      self.preds_p3 = tf.nn.bias_add(conv, biases, name='bias_add') #60*80*256*26*9
      #total macc 4.38G + 2.31G
      #param 1.43M + 1.96M
  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01, pool=False,
      freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """
    if pool:
      sq1x1 = self._conv_layer(
        layer_name+'/squeeze3x3', inputs, filters=s1x1, size=3, stride=2,
        padding='SAME', stddev=stddev, freeze=freeze)
    else:
      sq1x1 = self._conv_layer(
          layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
          padding='SAME', stddev=stddev, freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')