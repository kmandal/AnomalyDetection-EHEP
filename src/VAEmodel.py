import torch
import torch.nn as nn

class KinematicVAE(nn.Module):
    def __init__(self, input_dim: int = 3, latent_dim: int = 2, beta: float = 0.5):
        super().__init__()
        self.beta = beta
        
        # Encoder: compresses [log(pT), log(MET), mT] -> latent parameters (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        # Decoder: reconstructs features from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        recon_loss = torch.mean((x - x_recon) ** 2, dim=1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(recon_loss + self.beta * kl_loss)
