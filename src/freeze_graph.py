import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from config import *
from nets import *
import os
import uff
import torch
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'demo_net', 'zynqDet', """Neural net architecture.""")

tf.app.flags.DEFINE_string(
    'checkpoint', '/tmp/logs/+zynqDet/train/model_11999.ckpt-11999',
    """Path to the model parameter file.""")

tf.app.flags.DEFINE_string(
    'meta', '/tmp/logs/+zynqDet/train/model_11999.ckpt-11999.meta',
    """Path to the model parameter file.""")

tf.app.flags.DEFINE_string(
    'pbtxt', '/tmp/logs/+yolo+224/train/tensorflowModel_demo.pbtxt',
    """Path to the model parameter file.""")

if FLAGS.demo_net == 'yolo':
    mc = kitti_vgg16_config()
    mc.BATCH_SIZE = 1
    mc.LOAD_PRETRAINED_MODEL = False
    model = tinyDarkNet_FPN(mc, 0)
if FLAGS.demo_net == "zynqDet":
    mc = kitti_zynqDet_FPN_config()
    mc.BATCH_SIZE = 1
    mc.LOAD_PRETRAINED_MODEL = False
    model = ZynqDet_FPN(mc, 0)

saver = tf.train.Saver(model.model_params)
# graph = tf.get_default_graph()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#     # saver = tf.train.import_meta_graph(FLAGS.meta)
    saver.restore(sess, FLAGS.checkpoint)
    tensor_list = {}
    for param in model.model_params:
        # print(param.name)
        tensor = tf.get_default_graph().get_tensor_by_name(param.name)
        tensor_list['/'.join(param.name.split('/')[1:])] = sess.run(tensor)
        if "kernels:0" in param.name or "kernel:0" in param.name:
        	tensor_list['/'.join(param.name.split('/')[1:])] = np.transpose(sess.run(tensor), [3,2,0,1])



    # for key, val in tensor_list.iteritems():
    #     print(key, val.shape)
    # print(tensor_list['conv1/biases:0'])

    # torch.save(tensor_list, "./data/ZynqNet/zynqnet_fpn.pkl")

# freeze_graph.freeze_graph(FLAGS.pbtxt, "", False, 
#                           FLAGS.checkpoint, "probability/final_class_prob_concat,IOU/det_boxes_concat",
#                            "save/restore_all", "save/Const:0",
#                            '/tmp/logs/+yolo+224/train/frozentensorflowModel.pb', True, "" )


# with tf.gfile.GFile('/tmp/logs/+yolo+224/train/frozentensorflowModel.pb', 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())

# print('Check out the input placeholders:')
# nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
# for node in nodes:
#             print(node)
# inputGraph = tf.GraphDef()
# with tf.gfile.Open('/tmp/logs/+yolo+224/train/frozentensorflowModel.pb', "rb") as f:
#     data2read = f.read()
#     inputGraph.ParseFromString(data2read)

# outputGraph = optimize_for_inference_lib.optimize_for_inference(
#                 inputGraph,
#                 ["batch/fifo_queue"], #an array of input nodes
#                 ["probability/final_class_prob_concat","IOU/det_boxes_concat"], # an array of output nodes
#                 tf.int32.as_datatype_enum)

# f = tf.gfile.FastGFile('/tmp/logs/+yolo+224/train/OptimizedGraph.pb', "w")
# f.write(outputGraph.SerializeToString())

# uff.from_tensorflow_frozen_model(frozen_file='/tmp/logs/+yolo+224/train/frozentensorflowModel.pb', 
#                     output_filename='/tmp/logs/+yolo+224/train/rt_model',
#                     output_nodes=["IOU/det_boxes_concat"], text=True)