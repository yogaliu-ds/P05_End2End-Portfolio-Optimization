import torch
import torch.nn as nn
import torch.nn.functional as F

class PortfolioBlock(nn.Module):
    def __init__(self):
        super(PortfolioBlock, self).__init__()
        
    def forward(self, scores):
        """
        Module 1: Basic Portfolio Block
        Input: scores [B, N]
        Output: weights [B, N]
        """
        # Apply softmax to get weights that sum to 1
        weights = F.softmax(scores, dim=-1)
        return weights
