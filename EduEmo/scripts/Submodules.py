import torch
import torch.nn as nn
import torch.nn.functional as F



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v,pre_attn=None, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) #256,8,6,6

        if mask is not None: # 256*1*6*6
            attn = attn.masked_fill(mask == 0, -1e9)

        if pre_attn is not None:

            attn = self.dropout(F.softmax(attn+pre_attn, dim=-1))
        else:
            attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
