import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import gather_operation, furthest_point_sample


def testtt():
    Input=torch.randn(1, 1024, 3).cuda()
    output=furthest_point_sample(Input, 512)
    print(output.shape)