import torch
import torch.nn as nn


class HallucinatorLoss(nn.Module):
    def __init__(self, num_classes, no_selectors = 8):
        super(HallucinatorLoss, self).__init__()
        self.num_classes = num_classes
        self.no_selectors = no_selectors

    def forward(self, values_memory):
        return torch.sum(1 - torch.sum(torch.sort(values_memory, dim=1)[0][:, -self.no_selectors:], dim=1))
