import torch
import torch.nn as nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        targets_onehot = inputs.new_zeros(inputs.size(0), inputs.size(1))
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_onehot, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets_onehot, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def create_loss ():
    print('Loading Focal Loss.')
    return FocalLoss(logits=True)