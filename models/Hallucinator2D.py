import torch
import torch.nn as nn


class Hallucinator2D(nn.Module) :

    def __init__(self, num_classes, feat_dim) :
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.hallucinator = nn.Sequential()
        self.hallucinator.add_module()