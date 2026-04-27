import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OCC_loss(nn.Module):
    
    def __init__(self, ord=2,):
        super().__init__()
        self.ord = ord
    def forward(self, z_c, center, eval=False):

        # z = z_c[0]-z_c[1]
        # diffs = torch.pow(z.norm(p=2,dim=-1),self.ord)
        diffs = torch.pow((z_c - center).norm(p=self.ord, dim=-1), self.ord)
        if eval:

            return diffs
        else:
            return diffs
