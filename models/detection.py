"""
Preferred detection models
-----------------------------------------------
(Segmentation-based Text Detection)
1.DB algorithm (Differentiable Binarization)
2. DB++ algorithm
-----------------------------------------------
(Regression-based Text Detection)
1. TextBoxes algorithm
2. CTPN (Connectionist Text Proposal Network)
"""
import numpy as np
import torch.nn as nn


class DB(nn.Module):
    def __init__(self, params):
        super(DB, self).__init__()
        pass

    def forward(self, x):
        pass


def find_conv2d_out_shape(h_in, w_in, conv, pool=2):
    """
    :param h_in: an integer representing the height of input data
    :param w_in: an integer representing the width of input data
    :param conv:  an object of the CNN layer
    :param pool: an integer representing the pooling size and default to 2
    :return:
    """
    # get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    # Ref: https://pytorch.org/docs/stable/nn.html
    h_out = np.floor((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w_out = np.floor((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    if pool:
        h_out /= pool
        w_out /= pool
    return int(h_out), int(w_out)
