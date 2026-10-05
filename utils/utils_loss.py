import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    
    def __init__(self, weight, reduction='mean'):
        super(CELoss, self).__init__()
        self.celoss = torch.nn.CrossEntropyLoss(weight, reduction)

    def forward(self, logpt, targets):

        logpt = logpt.contiguous().view(-1, logpt.shape[-1]) # [batch* 4096, C]
        targets = targets.contiguous().view(-1) # [batch* 4096]

        loss = self.celoss(logpt,targets)

        return loss
