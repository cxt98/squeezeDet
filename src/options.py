"""
Options
"""
import argparse
import numpy as np


def parse_opts():
    """ options """
    parser = argparse.ArgumentParser(description='Various Paramaters')
    # training options
    parser.add_argument('--net', default='vgg16', type=str,
                        help='network name')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model from imagenet')
    parser.add_argument('--loss', default='ce',
                        choices=['bce', 'ce'], help='loss choice')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--train_batch', default=128,
                        type=int, help='training batch size')
    parser.add_argument('--test_batch', default=16,
                        type=int, help='testing batch size')
    parser.add_argument('--num_cls', default=22, type=int,
                        help='number of classes')
    parser.add_argument('--momentum', default=0.1, type=float, help='momentum')
    parser.add_argument('--w_decay', default=5e-4,
                        type=float, help='weight decay')
    parser.add_argument('--max_epoch', default=200, type=int,
                        help='maximum number of epoches')
    parser.add_argument('--test_iter', default=3, type=int,
                        help='test every n epoches')
    parser.add_argument('--resume', type=int, default=0,
                        help='resume from checkpoint')
    parser.add_argument('--img_size', default=224, type=int,
                        help='training image size (receptive field size)')
    parser.add_argument('--cls_loss', default=1.0,
                        type=float, help='classification loss coefficient')
    parser.add_argument('--shp_loss', default=0.0,
                        type=float, help='shape loss coefficient')
    parser.add_argument('--logdir', default='', type=str,
                        help='log directory')

    # testing options
    parser.add_argument('--scale', default=1, type=int,
                        help='image scale when testing')
    parser.add_argument('--test_cls', default='tide', type=str,
                        help='test class name')
    parser.add_argument('--scene_name', default='', type=str,
                        help='test scene name')
    parser.add_argument('--heat_pad', default=4, type=int,
                        help='heatmap padding')
    parser.add_argument('--cls_thresh', default=0.5, type=float,
                        help='classifier threshold')
    parser.add_argument('--ov_thresh', default=0.5, type=float,
                        help='overlapping threshold')
    parser.add_argument('--low_thresh', default=0.0, type=float,
                        help='lower bound of the threshold')
    parser.add_argument('--benchmark', action='store_true',
                        help='testing on the test set')

    # general
    parser.add_argument('--stride', default=32, type=int,
                        help='bbox stride from nn maxpool')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='verbose mode')
    parser.add_argument('--viz_heat', action='store_true',
                        help='visualize heatmap')
    parser.add_argument('--viz_bbox', action='store_true',
                        help='visualize bounding boxes')
    parser.add_argument('--viz_pr', action='store_true',
                        help='visualize pr curve')
    parser.add_argument('--nms_thresh', default=0.0, type=float,
                        help='nms threshold')
    parser.add_argument('--nms_gpu_id', default=0,
                        type=int, help='NMS GPU id')
    parser.add_argument('--num_ppsl', default=100, type=int,
                        help='number of proposal for R-CNN')
    parser.add_argument('--eval_prefix', default='density', type=str,
                        help='evaluation prefix')
    parser.add_argument('--eval_dir', default='model_check', type=str,
                        help='evaluation directory')

    # misc
    parser.add_argument('--u_thresh', default=1.0, type=float,
                        help='upper threshold for whatever')
    parser.add_argument('--ohem_lth', default=0.3, type=float,
                        help='ohem lower threshold')
    parser.add_argument('--ohem_uth', default=0.7, type=float,
                        help='ohem upper threshold')
    parser.add_argument('--checkpoint', default="./checkpoint/ckpt_15_epsilon_1.pkl", type=str)


    #adversarial
    parser.add_argument('--eps', default=0.1, type=float,
                        help='--epsilon for fast gradeint attack')
    parser.add_argument('--norm', default="inf", type=str)
    parser.add_argument('--adv', action='store_true',
                        help='using adversarial training')
    parser.add_argument('--adv_train', action='store_true',
                        help='using adversarial training')
    parser.add_argument('--viz_adv', action='store_true', help='using visualization during training')

    parser.add_argument('--targeted', action='store_true', help='targeted adversarial training')
    # parse
    opts = parser.parse_args()

    # extra
    opts.save_heat = True
    opts.scale_pyramid = [0.75, 1.0, 1.5, 2.0, 2.5]
    # opts.scale_pyramid = [1]
    # opts.scale_pyramid = [0.5, 0.75, 1.0, 1.5, 2.0]
    # opts.scale_pyramid = [2.0]
    # opts.scale_pyramid = [1.0, 2.0, 4]
    opts.ap_r_list = [1] #[1, 1.5, 0sq.67]
    # w/h [2.75, 2.25, 1.75, 1.25, 0.75, 0.25] / label: [1 2 3 4 5 6], 0 for bg
    opts.ap_range = [2.5, 2, 1.5, 1, 0.5]
    opts.ap_label = [2.75, 2.25, 1.75, 1.25, 0.75, 0.25]
    opts.num_shape = len(opts.ap_label) + 1
    opts.train_img_mean = (0.48166848598, 0.382752909145, 0.378840050744)
    opts.train_img_std = (0.26058393234, 0.281072581418, 0.285972542306)
    opts.imagenet_img_mean = (0.485, 0.456, 0.406)
    opts.imagenet_img_std = (0.229, 0.224, 0.225)
    opts.train_img_mean = opts.imagenet_img_mean
    opts.train_img_std = opts.imagenet_img_std

    # opts.gt_path = '/home/liuyanqi/caffe/adversail_training/data/SingleScenes'
    # if not opts.adv:
    #     opts.gt_path='/home/liuyanqi/caffe/adversail_training/data/RGBD_Object_Dataset'
    # else:
        # opts.gt_path = '/home/liuyanqi/caffe/adversail_training/data/robust_obj_detect_adv_dataset-master'
        # opts.gt_path = '/home/liuyanqi/caffe/pyramid_cnn/data/extract/'
    opts.gt_path = '/home/liuyanqi/caffe/pyramid_cnn/data/adversarial/'

    opts.out_path = './data/output/'
    # opts.labels = {'downy': 5, 'toy': 14, 'blue_cup': 1,
    #                'coke': 3, 'ranch': 6, 'spray_bottle': 10,
    #                'sugar': 11, 'tide': 13,
    #                'detergent': 4, 'clorox': 2, 'background': 0,
    #                'scotch_brite': 9, 'red_bowl': 7, 'waterpot': 15,
    #                'sunscreen': 12, 'salt': 8}

    opts.labels =  {'006_mustard_bottle': 4, '061_foam_brick': 20, '025_mug': 13,
     '021_bleach_cleanser': 11, '051_large_clamp': 18, '035_power_drill': 14, 
     '024_bowl': 12, '005_tomato_soup_can': 3, '009_gelatin_box': 7, '004_sugar_box': 2, 
     '019_pitcher_base': 10, 'background': 21, '037_scissors': 16, '052_extra_large_clamp': 19,
      '040_large_marker': 17, '010_potted_meat_can': 8, '002_master_chef_can': 0, 
      '007_tuna_fish_can': 5, '036_wood_block': 15, '008_pudding_box': 6, '003_cracker_box': 1, 
      '011_banana': 9}

    # opts.label_list = ['background', 'blue_cup', 'clorox', 'coke', 'detergent', 'downy', 'ranch', 'red_bowl', 'salt', 'scotch_brite', 'spray_bottle', 'sugar', 'sunscreen', 'tide', 'toy', 'waterpot']
    opts.label_list = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick', 'background']

    if opts.num_cls != len(opts.labels): # label sanity check
        raise Exception('Number of training classes does not match.')

    opts.use_gpu_nms = True

    # odd for training, even for testing
    opts.train_set = range(1, 61, 2)
    opts.test_set = range(1, 41)
    # opts.test_set = list(set(range(2, 61, 2)) - set([6, 8]))
    opts.adversarial_test_set = list(set(range(1, 21)) - set([2, 3, 4, 5, 6]))
    opts.adversarial_test_variation = ['B', 'D', 'H', 'O1', 'O2', 'O3']
    opts.traintest = opts.train_set + opts.test_set

    opts.norm_dict = {
    "inf": np.inf,
    "2": 2
    }

    return opts
