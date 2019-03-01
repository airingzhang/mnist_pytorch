'''
Created on Feb 27, 2019

@author: airingzhang
'''
import torch.nn as nn
import torch.nn.functional as F
class Loss(nn.Module):
    # simple nll loss
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, output, target):
        return F.nll_loss(output, target)