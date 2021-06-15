import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)
