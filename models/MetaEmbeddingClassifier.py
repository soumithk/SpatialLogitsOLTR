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

        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)

        self.mask = torch.zeros(num_classes).cuda()
        self.epoch = 0
        
    def forward(self, x, centroids, *args):

        self.mask[:self.epoch * 25] = 1
        
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
        # values_memory = x.new_ones((batch_size, self.num_classes))
        # values_memory.fill_(0.001)
        # close_centroids_indices = torch.sort(dist_cur, 1)[1][:, :16]
        # idx = [
        #     torch.cuda.LongTensor(range(batch_size)).unsqueeze(1), 
        #     torch.cuda.LongTensor(close_centroids_indices) 
        # ]
        # values_memory2 = self.fc_hallucinator2(x.clone())
        # values_memory2 = F.softmax(values_memory2, dim=1)
        # print(values_memory2[0])

        # # bs * 1000, 1000 * 512 -> bs * 512
        # memory_feature = torch.matmul(values_memory, keys_memory)
        # # bs * 1 * 16 , bs * 16 * 512 ->  bs * 1 * 512
        # values_memory2 = values_memory2.unsqueeze(1)
        # memory_feature2 = torch.matmul(values_memory2, centroids_expand[idx])
        # memory_feature2 = memory_feature2.squeeze(1)
        # removed
        values_memory = self.fc_hallucinator(x.clone())
        # print(F.softmax(values_memory[:, :200], dim=1)[0])
        values_memory = F.softmax(values_memory, dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        # computing concept selector
        concept_selector = self.fc_selector(x.clone())
        concept_selector = concept_selector.tanh() 

        # EXPERIMENTAL
        # added
        # concept_selector2 = self.fc_selector2(x.clone())
        # concept_selector2 = concept_selector2.sigmoid()
        # print(direct_feature.shape, concept_selector.shape, memory_feature.shape, (concept_selector * memory_feature).shape, concept_selector2.shape, 
        #         memory_feature2.shape, (concept_selector2 * memory_feature2).shape)
        # x = reachability * (direct_feature + concept_selector * memory_feature + concept_selector2 * memory_feature2)
        # removed
        x = reachability * (direct_feature + concept_selector * memory_feature)

        # storing infused feature
        infused_feature = concept_selector * memory_feature 
        
        logits = self.cosnorm_classifier(x)

        return logits, [direct_feature, infused_feature]
    
    def set_epoch(self, epoch) :
        self.epoch = epoch
    
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