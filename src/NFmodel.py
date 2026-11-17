import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, mask_config=0, hidden_dim=64):
        super(AffineCouplingLayer, self).__init__()
        self.dim = dim
        self.mask_config = mask_config 
        
        # Generalized split for even and odd dimensions:
        # in_dim: number of features passed into the scale/translate networks
        # out_dim: number of features transformed (scaled & translated)
        self.in_dim = dim // 2
        self.out_dim = dim - self.in_dim
        
        self.scale_net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.out_dim),
            nn.Tanh()  # Bounds scale for numerical stability
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.out_dim)
        )

    def forward(self, x):
        batch_size, dim = x.shape
        
        # Generalized split based on mask configuration
        if self.mask_config == 0:
            x1, x2 = x[:, :self.in_dim], x[:, self.in_dim:]
        else:
            x1, x2 = x[:, self.out_dim:], x[:, :self.out_dim]
            
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        
        # Affine transformation
        y2 = x2 * torch.exp(s) + t
        
        if self.mask_config == 0:
            y = torch.cat([x1, y2], dim=1)
        else:
            y = torch.cat([y2, x1], dim=1)
            
        # Sum of log determinant of the Jacobian
        log_det_jacobian = torch.sum(s, dim=1)
        return y, log_det_jacobian

    def inverse(self, y):
        batch_size, dim = y.shape
        
        if self.mask_config == 0:
            y1, y2 = y[:, :self.in_dim], y[:, self.in_dim:]
        else:
            y1, y2 = y[:, self.out_dim:], y[:, :self.out_dim]
            
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        
        x2 = (y2 - t) * torch.exp(-s)
        
        if self.mask_config == 0:
            x = torch.cat([y1, x2], dim=1)
        else:
            x = torch.cat([x2, y1], dim=1)
            
        return x


class RealNVP(nn.Module):
    def __init__(self, dim=3, num_layers=6, hidden_dim=64):
        super(RealNVP, self).__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim=dim, mask_config=i % 2, hidden_dim=hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, x):
        """Maps data x to latent z and tracks log-determinant Jacobian."""
        z = x
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
            
        return z, log_det_total

    def log_prob(self, x):
        """Computes exact log-likelihood log p(x) under the flow."""
        z, log_det = self.forward(x)
        # Base distribution: Standard Normal N(0, I)
        base_log_prob = -0.5 * torch.sum(z ** 2 + torch.log(torch.tensor(2 * 3.141592653589793, device=x.device)), dim=1)
        return base_log_prob + log_det
