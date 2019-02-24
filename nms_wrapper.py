"""
NMS wrapper
"""
import numpy as np
from nms.gpu_nms import gpu_nms  # pylint: disable=E0611
from nms.cpu_nms import cpu_nms  # pylint: disable=E0611


def selective_nms(dets, opts):
    """ perform nms based on square bbox """
    square_dets = dets[::len(opts.ap_r_list)]  # skip to square bboxes
    square_keep = nms(square_dets.astype(np.float32), opts.nms_thresh,  # pylint: disable=E1101
                      opts)
    keep = []
    for sq_idx in square_keep:  # iter thru sq_idx
        keep.append(sq_idx * len(opts.ap_r_list))  # retrieve square index
        # interpolate keep with other anchors idx
        for extra_idx in range(1, len(opts.ap_r_list)):
            keep.append(sq_idx * len(opts.ap_r_list) + extra_idx)
    return keep


def nms(dets, thresh, opts, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if opts.use_gpu_nms and not force_cpu:
        return gpu_nms(dets, thresh, device_id=opts.nms_gpu_id)
    else:
        return cpu_nms(dets, thresh)
