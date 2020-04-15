# attention.py

import torch
from torch import nn
import math
import torch.nn.functional as F
from transformer_classifier.Model_Transformer.train_utils import clones
import numpy as np

def attention_original(query, key, value, mask=None, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('q',query.size())
    # print('s',scores.size())
    # print('m',mask.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # scores = scores.masked_fill(mask == 0, 0)
    p_attn = F.softmax(scores, dim = -1)


    # p_attn = p_attn.masked_fill(p_attn == 0.0100,0) ###
    # # print(p_attn.transpose(1, 2)[0][99])

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def attention(query, key, value, mask=None, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('q',query.size())
    # print('s',scores.size())
    # print('m',mask.size())
    if mask is not None:
        print(mask.size())
        print(scores.size())
        scores = scores.masked_fill(mask == 0, -1e9)
        # scores = scores.masked_fill(mask == 0, 0)
    p_attn = F.softmax(scores, dim = -1)


    # p_attn = p_attn.masked_fill(p_attn == 0.0100,0) ###
    # # print(p_attn.transpose(1, 2)[0][99])

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Multi-head attention"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k

        if mask is not None:
            mask = mask.view(nbatches,-1,self.h,self.d_k*2).transpose(1,2) ###

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]


        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # print(self.attn.transpose(1, 2)[0][99])
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
