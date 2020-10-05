import torch
import torch.nn as nn
from torch.nn import functional as F
from models.CosNormClassifier import CosNorm_Classifier
from utils import *

import pdb

class MetaEmbedding_Classifier(nn.Module):
    
    def __init__(self, feat_dim=2048, num_classes=1000):
        super(MetaEmbedding_Classifier, self).__init__()
        self.num_classes = num_classes
        # EXPERIMENTAL
        # added
        self.fc_hallucinator2 = nn.Linear(feat_dim, num_classes)
        self.fc_selector2 = nn.Linear(feat_dim, feat_dim)
        # removed
        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)

        self.mask = torch.zeros(num_classes).cuda()
        self.epoch = 0
        
    def forward(self, x, centroids, *args):
        
        # storing direct feature
        direct_feature = x.clone()

        batch_size = x.size(0)
        feat_size = x.size(1)
        
        # set up visual memory
        x_expand = x.clone().unsqueeze(1).expand(-1, self.num_classes, -1)
        centroids_expand = centroids.clone().unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids.clone()
        
        # computing reachability
        dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
        values_nn, labels_nn = torch.sort(dist_cur, 1)
        scale = 10.0
        reachability = (scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)

        # computing memory feature by querying and associating visual memory
        # EXPERIMENTAL
        # added
        values_memory = self.fc_hallucinator(x.clone())
        values_memory = F.softmax(values_memory, dim=1)
        values_memory2_ = self.fc_hallucinator2(x.clone())
        values_memory2 = F.softmax(values_memory2_, dim=1)

        memory_feature = torch.matmul(values_memory, keys_memory)
        memory_feature2 = torch.matmul(values_memory2_, keys_memory)
        # removed
        # values_memory = self.fc_hallucinator(x.clone())
        # values_memory = F.softmax(values_memory, dim=1)
        # values_memory.fill_(0.001)
        # memory_feature = torch.matmul(values_memory, keys_memory)

        # computing concept selector
        concept_selector = self.fc_selector(x.clone())
        concept_selector = - concept_selector.sigmoid() 

        # EXPERIMENTAL
        # added
        concept_selector2 = self.fc_selector2(x.clone())
        concept_selector2 = concept_selector2.sigmoid()
        # print(concept_selector[0] + concept_selector2[0])
        # print(direct_feature.shape, concept_selector.shape, memory_feature.shape, (concept_selector * memory_feature).shape, concept_selector2.shape, 
        #         memory_feature2.shape, (concept_selector2 * memory_feature2).shape)
        x = reachability * (direct_feature + concept_selector * memory_feature + concept_selector2 * memory_feature2)
        # removed
        # x = reachability * (direct_feature + concept_selector * memory_feature)

        # storing infused feature
        infused_feature = concept_selector * memory_feature + concept_selector2 * memory_feature2
        
        logits = self.cosnorm_classifier(x)

        return logits, [direct_feature, infused_feature, values_memory2]

    def set_epoch(self, epoch) :
        self.epoch = epoch
        self.mask[:(epoch * 25)] = 1
    
def create_model(feat_dim=2048, num_classes=1000, stage1_weights=False, dataset=None, test=False, *args):
    print('Loading Meta Embedding Classifier.')
    clf = MetaEmbedding_Classifier(feat_dim, num_classes)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            clf.fc_hallucinator = init_weights(model=clf.fc_hallucinator,
                                                    weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset,
                                                    classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf
