"""
Evaluate detection result, generate PR curve
"""

from __future__ import print_function
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import options
from nms_wrapper import nms, selective_nms
np.set_printoptions(threshold=np.inf, suppress=True)


def calc_ov(bb, bbgt):
    """ calculate overlapping area
        param: bb, bbgt
        return: ov, iw, ih
    """
    bi = np.array([np.maximum(bb[0], bbgt[0]),
                   np.maximum(bb[1], bbgt[1]),
                   np.minimum(bb[2], bbgt[2]),
                   np.minimum(bb[3], bbgt[3]), ])
    ih = bi[2] - bi[0] + 1
    iw = bi[3] - bi[1] + 1
    ov = float("-inf")
    if iw > 0 and ih > 0:
        # compute overlap as IoU
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) +\
            (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) -\
            iw * ih
        ov = 1.0 * iw * ih / ua
    return ov, iw, ih


def calc_ap(rec, prec):
    """ Calculate AP """
    mrec = np.hstack([0, rec, 1])
    mpre = np.hstack([0, prec, 0])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = np.maximum(mpre[i], mpre[i + 1])

    i = np.add(np.nonzero(np.not_equal(mrec[1:], mrec[:-1])), 1)
    return np.sum((mrec[i] - mrec[i - 1]) * mpre[i])


def read_gt(scene, opts):
    """ Read ground truth from txt file """
    img_path = os.path.join(opts.gt_path, scene, 'scenergb.jpg')
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


def plot_pr(prec, rec, ap, obj):
    """ plot precision and recall curve """
    plt.figure()
    plt.plot(rec, prec, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.01])
    plt.title('PR Curve, Obj={:s}, AP (AUC)={:0.2f}'.format(obj, ap))

    # plot f1 score
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))


def calc_pr_batch(dets, scene_list, opts):
    """ calculate precision and recall giving all bboxes """
    # iter thru all gt to find npos
    npos = 0
    gt_dict = {}
    gt_det = {}
    for scene in scene_list:  # iter thru each scene for ground truth
        # read gt
        gt_bbox, _ = read_gt(scene, opts)
        # increment npos
        npos += gt_bbox.shape[0]
        # save it in a dict
        gt_dict[scene] = gt_bbox
        gt_det[scene] = np.zeros(gt_bbox.shape[0])

    confidence = dets[:, -2]
    ids = dets[:, -1]

    # sort detections by decreasing confidence
    sc = np.sort(-confidence)
    si = np.argsort(-confidence)
    ids = ids[si]
    BB = dets[si, 0:4]

    # assign detections to ground truth objects
    nd = len(confidence)
    tp = np.zeros((nd, 1))
    fp = np.zeros((nd, 1))

    # iter thru each detection bbox
    for d in range(0, nd):
        # assign detection to ground truth object if any
        bb = BB[d]
        ovmax = float("-inf")
        jmax = -1
        # find the gt bbox given scene_id
        scene_id = scene_list[ids[d].astype(int)]
        gt_bbox = gt_dict[scene_id]
        for j in range(0, gt_bbox.shape[0]):
            bbgt = gt_bbox[j]
            bi = np.array([np.maximum(bb[0], bbgt[0]),
                           np.maximum(bb[1], bbgt[1]),
                           np.minimum(bb[2], bbgt[2]),
                           np.minimum(bb[3], bbgt[3]), ])
            ih = bi[2] - bi[0] + 1
            iw = bi[3] - bi[1] + 1
            if iw > 0 and ih > 0:
                # compute overlap as IoU
                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) +\
                    (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) -\
                    iw * ih
                ov = 1.0 * iw * ih / ua
                if ov > ovmax:
                    ovmax = ov
                    jmax = j

        # assign detection as tp/fp
        # print(ovmax, opts.ov_thresh)
        if ovmax >= opts.ov_thresh:
            if gt_det[scene_id][jmax] == 0:
                tp[d] = 1  # true positive
                gt_det[scene_id][jmax] = 1  # assign true if found
            else:  # false positive (multiple detection)
                fp[d] = 1
        else:
            fp[d] = 1  # false positive

    # compute precision/recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = np.divide(tp, npos + np.finfo(float).eps)
    prec = np.divide(tp, fp + tp + np.finfo(float).eps)
    return prec, rec


def eval_batch(scene_list, opts):
    """ evaluate a batch of bbox files """
    mAP = 0
    n_obj = 0
    if opts.scene_name != '':  # eval one scene
        scene_list = [opts.scene_name]
    # opts.labels = ['scotch_brite']
    for obj in opts.labels:  # iter thru each obj
        if obj == 'background':  # skip background class
            continue
        opts.test_cls = obj
        dets_all = np.empty((0, 6))
        scene_id = 0
        for scene in scene_list:  # iter thru each scene
            # read bbox file
            dets = np.genfromtxt(opts.eval_dir + '/' +
                                 opts.eval_prefix + '_' + scene + '_' +
                                 opts.test_cls + '.txt')

            # remove the trash
            trash_idx = np.nonzero(dets < 0)[0][::2]
            dets = np.delete(dets, trash_idx, axis=0)
            # append scene_id to dets
            if dets.ndim > 1:
                dets = np.hstack(
                    [dets, scene_id * np.ones((dets.shape[0], 1))])
            else:
                dets = np.hstack([dets, scene_id])

            # use nms to prune dets
            if opts.nms_thresh > 0.0:
                # keep = selective_nms(dets, opts)
                keep = nms(dets.astype(np.float32), opts.nms_thresh, opts) # pylint: disable=E1101
                nms_dets = dets[keep, :]
                dets_all = np.vstack([dets_all, nms_dets])  # adding dets
            else:
                dets_all = np.vstack([dets_all, dets])  # adding dets

            scene_id += 1  # increment scene_id


        # filter low likelihood bbox
        if opts.low_thresh > 0.0:
            low_idx = np.nonzero(dets_all[:, 4] < opts.low_thresh)
            dets_all = np.delete(dets_all, low_idx, axis=0)
        # if obj =="red_bowl":
        #     print(dets_all)
        prec, rec = calc_pr_batch(dets_all, scene_list, opts)
        ap = calc_ap(rec, prec)
        # print('Eval: {}, AP: {}, precision: {}, recall: {}'.format(obj, ap, np.mean(prec), np.mean(rec)))
        print(ap)
        mAP += ap
        n_obj += 1
        if opts.viz_pr:
            plot_pr(prec, rec, ap, opts.test_cls)

    mAP = mAP / n_obj
    print('mAP: {}'.format(mAP))
    if opts.viz_pr:
        plt.show()

    return mAP


if __name__ == "__main__":
    opts = options.parse_opts()
    # scene_list = ['exp{:02d}'.format(i) for i in opts.train_set]
    scene_list = ['exp{:02d}'.format(i) for i in opts.test_set]
    if opts.adv:
        scene_variation = ["B"]
        # scene_variation = ["B", "D", "H", "L", "O1", "O2", "O3"]
        # scene_variation = ["D"]
        # scene_variation = ["H", "L"]
        # scene_variation = ["O1", "O2", "O3"]
        # scene_variation = ["D"]
        scene_list = []
        for v in scene_variation:
            for i in opts.adversarial_test_set:
                scene_list.append('exp{:03d}_{}'.format(i, v))
    print(scene_list)
    eval_batch(scene_list, opts)
