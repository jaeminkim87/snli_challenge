import torch
import copy
import torch.nn as nn
from torch.nn.modules.container import ModuleList


def _reset_parameters(self):
    for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
