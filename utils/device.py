

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,0"

__device = torch.device('cuda:0')


def set(d):
    global __device
    __device = d

def get():
    return __device



