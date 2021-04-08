import torch
from torch import nn
import torch.nn.functional as F

from math import pi



class WaveAttn(nn.Module):
    def __init__(self, ch_in=3, ch_out=100):
        super(WaveAttn, self).__init__()

        self.conv = nn.Conv2d(ch_in, 1, 1, 1, 0, bias=False)
        self.register_buffer('dots', torch.linspace(-6*pi, 6*pi, ch_out).view(1,ch_out,1,1).float() )

    def forward(self, tensor):
        freq = self.conv(tensor)
        return torch.sin(freq*self.dots)






class SpatialNorm(nn.Module):
    def __init__(self, ch_in=64, n_set=8):
        super(SpatialNorm, self).__init__()

        self.n_set = n_set

        self.mu = nn.ParameterList([
                        nn.Parameter(torch.zeros(1,ch_in,1,1, requires_grad=True))
                        for _ in range(n_set*2) ])
        self.sigma = nn.ParameterList([
                        nn.Parameter(torch.ones(1,ch_in,1,1, requires_grad=True))
                        for _ in range(n_set*2) ])

        self.conv_map = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(ch_in, 1, 3, 1, 1, bias=False),
                            nn.BatchNorm2d(1))
                        for _ in range(n_set) ])

    def forward(self, feat_org):
        feat_new = torch.empty_like(feat_org)
        for idx in range(self.n_set):
            map_0 = self.conv_map[idx](feat_org)
            feat_new += map_0 * (self.sigma[idx] * feat_org + self.mu[idx])
            #map_1 = 1 - map_0
            #new_feat += map_1 * (self.sigma[idx+self.n_set] * feat +\
            #                         self.mu[idx+self.n_set])
        return feat_org + feat_new
