import torch
import torch.nn as nn
from torch.nn import functional as F
# from models.CosNormClassifier import CosNorm_Classifier
# from utils import *

import pdb

# layer 1  torch.Size([128, 64, 56, 56])
# layer 2  torch.Size([128, 128, 28, 28])
# layer 3  torch.Size([128, 256, 14, 14])
# layer 4  torch.Size([128, 512, 7, 7])

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SpatialLogitsClassifier(nn.Module):

    def __init__(self, feat_dim=2048, num_classes=1000):
        super(SpatialLogitsClassifier, self).__init__()
        self.num_classes = num_classes

        # branch 3
        self.GAP_Classifier = nn.Linear(feat_dim, num_classes)

        # branch 2
        self.branch2_pool1 = nn.AvgPool2d(7, stride=8)
        self.branch2_conv1 = conv3x3(64, 32)
        self.branch2_pool2 = nn.AvgPool2d(7, stride=4, padding=2)
        self.branch2_conv2 = conv3x3(128, 64)
        self.branch2_pool3 = nn.AvgPool2d(8, stride=1)
        self.branch2_conv3 = conv3x3(256, 128)
        self.branch2_conv4 = conv3x3(512, 256)
        # 32 + 64 + 128 + 256
        self.spatial_logits_generator = nn.Sequential(
            nn.BatchNorm2d(32 + 64 + 128 + 256),
            nn.ReLU(),
            conv3x3(32 + 64 + 128 + 256, self.num_classes)
        )

        # branch 1
        self.attention_map_generator = nn.Sequential(
            conv3x3(512, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            conv3x3(128, 64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x, feature_maps, *args):
        """
        x (bs, 512)
        featuremaps

        returns  
        gap_logits (bs, 1000)
        spatial_logits (bs, 1000)
        """

        assert len(feature_maps) == 4

        gap_logits = self.GAP_Classifier(x)
        # gap_logits = F.softmax(gap_logits, dim = 1)

        x1 = self.branch2_conv1(self.branch2_pool1(feature_maps[0]))
        x2 = self.branch2_conv2(self.branch2_pool2(feature_maps[1]))
        x3 = self.branch2_conv3(self.branch2_pool3(feature_maps[2]))
        x4 = self.branch2_conv4(feature_maps[3])
        spatial_feature_map = torch.cat([x1, x2, x3, x4], dim=1)
        spatial_logits = self.spatial_logits_generator(spatial_feature_map)
        spatial_logits = F.softmax(spatial_logits, dim = 1)

        attention_map = self.attention_map_generator(feature_maps[3])
        bs, c, h, w = attention_map.size()
        attention_map = F.softmax(attention_map.view(bs, -1), dim=1).view(bs, c, h, w)

        spatial_logits = spatial_logits * attention_map.repeat(1, self.num_classes, 1, 1)
        spatial_logits = torch.sum(spatial_logits.view(bs, self.num_classes, -1), dim=2)

        return gap_logits, spatial_logits




def create_model(feat_dim=512, num_classes=1000, stage1_weights=False, dataset=None, test=False, *args):
    print('Loading Meta Embedding Classifier.')
    clf = SpatialLogitsClassifier(feat_dim, num_classes)
    return clf


if __name__ == "__main__" :

    module = SpatialLogitsClassifier(512, 1000).cuda()
    a, b = module(torch.rand(128, 512).cuda(), [torch.rand(128, 64, 56, 56).cuda(), torch.rand(128, 128, 28, 28).cuda(), 
                                            torch.rand(128, 256, 14, 14).cuda(), torch.rand(128, 512, 7, 7).cuda()])
    print(a.shape, b.shape)
