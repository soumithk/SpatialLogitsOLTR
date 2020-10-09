import torch
import torch.nn as nn
from torch.nn import functional as F


class SoalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

        self.gap_cross_entropy = nn.CrossEntropyLoss()
        self.kl_divergence = nn.KLDivLoss()

        self.spatial_cross_entropy = nn.CrossEntropyLoss()

    def forward(self, gap_logits, spatial_logits, targets):

        self.loss_sl =  self.spatial_cross_entropy(spatial_logits, targets)

        self.loss_dl = self.kl_divergence(F.log_softmax(spatial_logits), F.softmax(gap_logits)) + \
            0.5 * self.gap_cross_entropy(gap_logits, targets)

        return self.loss_sl + self.loss_dl

def create_loss ():
    print('Loading SAOL Loss.')
    return SoalLoss(logits=True)