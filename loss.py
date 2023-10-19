import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ASDLoss(nn.Module):
    def __init__(self, num_classes, w_normal=1):
        super(ASDLoss, self).__init__()
        weight = torch.ones((num_classes,)).float()
        weight[0] = float(w_normal)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, out, target):
        ce_loss = self.ce_loss(out, target)
        return ce_loss
