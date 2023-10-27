import torch
from torch import nn
import numpy as np

class CouplingLayer(nn.Module):
    def __init__(self, mask_type, input_dim):
        super(CouplingLayer, self).__init__()
        self.mask_type = mask_type

        self.s = nn.Sequential(
            nn.Linear(input_dim//2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim//2),
            nn.Tanh()
        )

        self.t = nn.Sequential(
            nn.Linear(input_dim//2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim//2),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        if self.mask_type == 'right':
            y = torch.cat([x1, x2 * torch.exp(self.s(x1)) + self.t(x1)], dim=-1)
            logdet = torch.sum(self.s(x1), dim=-1)
        if self.mask_type == 'left':
            y = torch.cat([x1 * torch.exp(self.s(x2)) + self.t(x2), x2], dim=-1)
            logdet = torch.sum(self.s(x2), dim=-1)
        return y, logdet
    
    def reverse(self, y):
        y1 = y[..., :y.shape[-1]//2]
        y2 = y[..., y.shape[-1]//2:]
        if self.mask_type == 'right':
            x = torch.cat([y1, (y2 - self.t(y1)) * torch.exp(-self.s(y1))], dim=-1)
        if self.mask_type == 'left':
            x = torch.cat([(y1 - self.t(y2)) * torch.exp(-self.s(y2)), y2], dim=-1)
        return x, torch.sum(self.s(y1), dim=-1)

class realNVP(nn.Module):
    def __init__(self):
        super(realNVP, self).__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(mask_type='right', input_dim=2),
            CouplingLayer(mask_type='left', input_dim=2),
            CouplingLayer(mask_type='right', input_dim=2),
            CouplingLayer(mask_type='left', input_dim=2),
            CouplingLayer(mask_type='right', input_dim=2),
            CouplingLayer(mask_type='left', input_dim=2),
            CouplingLayer(mask_type='right', input_dim=2),
            CouplingLayer(mask_type='left', input_dim=2),
        ])

    def forward(self, x):
        log_det = 0
        for layer in self.layers:
            x, ldet = layer(x)
            log_det += ldet
        return x, log_det

    def reverse(self, y):
        log_det = 0
        for layer in reversed(self.layers):
            y, ldet = layer.reverse(y)
            log_det += ldet
        return y, log_det