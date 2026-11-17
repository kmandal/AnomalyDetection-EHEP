import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from VAEmodel import KinematicVAE 
from NFmodel import RealNVP

def main():
    parser = argparse.ArgumentParser(description="Unified EHEP Model Training (VAE or NF)")
    parser.add_argument('--model', type=str, default='nf', choices=['vae', 'nf'], help="Model type to train")
    args = parser.parse_args()

    print(f"Loading EHEP training features for {args.model.upper()}...")
    df_train = pd.read_csv("data/sm_train.csv")
    features = df_train[["log_MET", "log_pT_lep", "mT"]].values
    weights = df_train["weight"].values

    # Compute and save standard scaling parameters
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    os.makedirs("data", exist_ok=True)
    np.save("data/scaler_mean.npy", mean)
    np.save("data/scaler_std.npy", std)

    features_scaled = (features - mean) / std

    dataset = TensorDataset(
        torch.tensor(features_scaled, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Initialize model based on argument
    if args.model == 'vae':
        model = KinematicVAE(input_dim=3, latent_dim=2, beta=0.5)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 10
    else:
        model = RealNVP(dim=3, num_layers=6, hidden_dim=64)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 10

    print(f"Training {args.model.upper()} on Standard Model background...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            
            if args.model == 'vae':
                x_recon, mu, logvar = model(x)
                loss = model.loss_function(x, x_recon, mu, logvar)
            else:
                log_p = model.log_prob(x)
                loss = -torch.mean(log_p)  # Negative Log-Likelihood
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

    weight_path = f"data/{args.model}_model.pth"
    torch.save(model.state_dict(), weight_path)
    print(f"Model weights saved successfully to {weight_path}")

if __name__ == "__main__":
    main()
