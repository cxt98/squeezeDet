import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import math
import zynqnet_fpn
import os
import cv2
import numpy as np
import utils as util
import tensorflow as tf
import sys

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CLASSES = 21
EXP_THRESH = 1.0

def set_anchors(H, W):
  B = 1
  ANCHOR_SHAPES = np.array([[224, 224]])

  anchor_shapes = np.reshape([[ANCHOR_SHAPES]]*H*W, (H,W,B,2))

  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return torch.from_numpy(anchors).float().cuda()

def interpret_graph(preds, ANCHORS):
	# set_anchors(mc, scale)
	print("mc.ANCHORS", ANCHORS)
	num_class_probs = 21
	pred_class_probs = torch.reshape(
	    torch.nn.Softmax(dim=1)(
	        torch.reshape(
	            preds[:, :,  :, :num_class_probs],
	            (-1, num_class_probs)
	        )
	    ),
	    (1, ANCHORS, CLASSES)
	)
	# print(pred_class_probs.shape)

	# confidence
	num_confidence_scores = 1+num_class_probs
	pred_conf = torch.sigmoid(
	    torch.reshape(
	        preds[:, :, :, num_class_probs:num_confidence_scores],
	        (1, ANCHORS)
	    )
	)

	# bbox_delta
	pred_bbox_delta = torch.reshape(
	    preds[:, :, :, num_confidence_scores:],
	    (1, ANCHORS, 4)
	)
	return pred_class_probs, pred_conf, pred_bbox_delta

def define_bbox(pred_bbox_delta, ANCHOR_BOX):
	delta_x, delta_y, delta_w, delta_h = torch.unbind(
	  pred_bbox_delta, dim=2)
	# set_anchors(mc, scale)

	anchor_x = ANCHOR_BOX[:, 0]
	anchor_y = ANCHOR_BOX[:, 1]
	anchor_w = ANCHOR_BOX[:, 2]
	anchor_h = ANCHOR_BOX[:, 3]

	box_center_x =  anchor_x + delta_x * anchor_w
	box_center_y =  anchor_y + delta_y * anchor_h
	# box_width = anchor_w * util.safe_exp(delta_w, EXP_THRESH)
	# box_height = anchor_h * util.safe_exp(delta_h, EXP_THRESH)
	box_width = anchor_w * torch.exp(delta_w)
	box_height = anchor_h * torch.exp(delta_h) # ok, this needs to be done on CPU side

	xmins, ymins, xmaxs, ymaxs = util.bbox_transform(
	    [box_center_x, box_center_y, box_width, box_height])

	xmins = xmins.cpu().detach().numpy()
	ymins = ymins.cpu().detach().numpy()
	xmaxs = xmaxs.cpu().detach().numpy()
	ymaxs = ymaxs.cpu().detach().numpy()

	# The max x position is mc.IMAGE_WIDTH - 1 since we use zero-based
	# pixels. Same for y.
	xmins = np.minimum(
	    np.maximum(0.0, xmins), IMAGE_WIDTH-1.0)

	ymins = np.minimum(
	    np.maximum(0.0, ymins), IMAGE_HEIGHT-1.0)

	xmaxs = np.maximum(
	    np.minimum(IMAGE_WIDTH-1.0, xmaxs), 0.0)

	ymaxs = np.maximum(
	    np.minimum(IMAGE_HEIGHT-1.0, ymaxs), 0.0)

	det_boxes = torch.transpose(
	    torch.stack(util.bbox_transform_inv(torch.FloatTensor([xmins, ymins, xmaxs, ymaxs]))),
	    1, 2) # this is not needed for hardware implementation
	return det_boxes

def get_probability(pred_class_probs, pred_conf, ANCHORS):
	# print("pred_class_prob", pred_class_probs.shape, "pred_conf: ", pred_conf.shape)
	probs = torch.mul(
	    pred_class_probs,
	    torch.reshape(pred_conf, (1, ANCHORS, 1)),
	)
	return probs
      
gt_path = '/home/liuyanqi/caffe/pyramid_cnn/data/adversarial'
scene_name = 'exp001_B'
net = zynqnet_fpn.zynqnet_fpn(pretrained=True).cuda()
img_path = os.path.join(gt_path, scene_name, 'scenergb.jpg')
im = cv2.imread(img_path)
BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])

input_image = im - BGR_MEANS

input_image = np.transpose(input_image, [2, 0, 1])
input_image = input_image.astype(np.float32, copy=False)
input_image = torch.from_numpy(input_image).cuda()
input_image = torch.unsqueeze(input_image, 0)


print(input_image[0,:,0,0])
c1, preds, preds_p4, preds_p3 = net(input_image)
print(c1.size())
print(c1.cpu().data.numpy()[0,:,1,1])
print(preds.size())
print(preds.cpu().data.numpy()[0,22:,:,:])
preds = preds.permute(0, 2, 3, 1)
print(preds[0,:,:,22:])
# print(preds_p4.shape)
# print(preds_p4.cpu().data.numpy()[0,:,1,0])
# print(preds_p3.shape)
# print(preds_p3.cpu().data.numpy()[0,:,1,0])


pred_class_probs1, pred_conf1, pred_bbox_delta1 = interpret_graph(preds, 300)
# pred_class_probs2, pred_conf2, pred_bbox_delta2 = interpret_graph(preds_p4, 1200)
# pred_class_probs3, pred_conf3, pred_bbox_delta3 = interpret_graph(preds_p3, 4800)

det_boxes1 = define_bbox(pred_bbox_delta1, set_anchors(15,20))
# det_boxes2 = define_bbox(pred_bbox_delta2, set_anchors(30,40))
# det_boxes3 = define_bbox(pred_bbox_delta3, set_anchors(60,80))

probs1 = get_probability(pred_class_probs1, pred_conf1, 300)
# probs2 = get_probability(pred_class_probs2, pred_conf2, 1200)
# probs3 = get_probability(pred_class_probs3, pred_conf3, 4800)

# print(probs1.shape)
# print(probs1.cpu().data.numpy()[0,0,:])
# print(np.sum(probs1.cpu().data.numpy()[0,:,0]))
print(pred_class_probs1.shape)
print(pred_class_probs1[0,0,:].cpu().data.numpy())
# print(np.sum(pred_class_probs1[0,0,:].cpu().data.numpy()))

print(pred_conf1.shape)
print(pred_conf1[0,:].cpu().data.numpy())

print(pred_bbox_delta1.shape)
print(pred_bbox_delta1[0,0,:].cpu().data.numpy())

print(probs1.shape)
print(probs1[0,0,:].cpu().data.numpy())

print(det_boxes1.shape)
print(det_boxes1[:,0,0])