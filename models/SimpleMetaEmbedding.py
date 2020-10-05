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
        self.fc_hallucinator = nn.Linear(feat_dim, 16)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)

        self.memory = nn.Parameter(torch.randn(16, feat_dim))
        
    def forward(self, x, *args):
        
        # storing direct feature
        direct_feature = x.clone()

        batch_size = x.size(0)
        feat_size = x.size(1)
        
        # set up visual memory
        x_expand = x.clone().unsqueeze(1).expand(-1, self.num_classes, -1)
        keys_memory = self.memory
        

        # computing memory feature by querying and associating visual memory
        values_memory = self.fc_hallucinator(x.clone())
        values_memory = F.softmax(values_memory, dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        # computing concept selector
        concept_selector = self.fc_selector(x.clone())
        concept_selector = concept_selector.tanh() 

        x =  (direct_feature + concept_selector * memory_feature)

        # storing infused feature
        infused_feature = concept_selector * memory_feature
        
        logits = self.cosnorm_classifier(x)

        return logits, [direct_feature, infused_feature]
    
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
