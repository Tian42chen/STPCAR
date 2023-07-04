import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import gather_operation, furthest_point_sample


def testtt():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Input=torch.randn(1, 1024, 3).to(device)
    output=furthest_point_sample(Input, 512)
    print(output.shape)