import os
import argparse
import torch
import pandas as pd
import numpy as np
import uproot

from VAEmodel import KinematicVAE
from NFmodel import RealNVP

def load_signal_from_root(file_path, tree_name="Events", mean=None, std=None):
    print(f"Loading signal file: {file_path}...")
    with uproot.open(file_path) as f:
        tree = f[tree_name]
        met = tree["MET"].array(library="np")
        pt_lep = tree["LeppT"].array(library="np")
        mt = tree["LepMT"].array(library="np")
        
    log_met = np.log(met + 1e-5)
    log_pt_lep = np.log(pt_lep + 1e-5)
    features = np.column_stack([log_met, log_pt_lep, mt])
    features_scaled = (features - mean) / std
    return torch.tensor(features_scaled, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Unified EHEP Model Evaluation")
    parser.add_argument('--model', type=str, default='nf', choices=['vae', 'nf'], help="Model type to evaluate")
    args = parser.parse_args()

    print(f"Running Evaluation for {args.model.upper()}...")
    
    mean = np.load("data/scaler_mean.npy")
    std = np.load("data/scaler_std.npy")

    # Load respective model
    if args.model == 'vae':
        model = KinematicVAE(input_dim=3, latent_dim=2, beta=0.5)
    else:
        model = RealNVP(dim=3, num_layers=6, hidden_dim=64)
        
    weight_path = f"data/{args.model}_model.pth"
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()

    # Load SM Test Split
    df_test = pd.read_csv("data/sm_test.csv")
    sm_test_features = df_test[["log_MET", "log_pT_lep", "mT"]].values
    sm_test_scaled = (sm_test_features - mean) / std
    sm_tensor = torch.tensor(sm_test_scaled, dtype=torch.float32)

    # Load Signal ROOT file
    signal_file = "data/VarTree_Sig_Prompt_500_420_full.root"
    signal_tensor = load_signal_from_root(signal_file, "Events", mean, std)

    with torch.no_grad():
        if args.model == 'vae':
            # VAE Evaluation (Reconstruction Error)
            x_recon_sm, _, _ = model(sm_tensor)
            sm_scores = torch.mean((sm_tensor - x_recon_sm) ** 2, dim=1)
            mean_sm = torch.mean(sm_scores).item()

            x_recon_sig, _, _ = model(signal_tensor)
            sig_scores = torch.mean((signal_tensor - x_recon_sig) ** 2, dim=1)
            mean_sig = torch.mean(sig_scores).item()

            print(f"\n[VAE Sanity Check] Mean SM Test Reconstruction Error: {mean_sm:.4f}")
            print(f"[VAE Signal Test]  Mean Stop Signal Reconstruction Error: {mean_sig:.4f}")
            print(f"-> Separation indicator: Signal score is {mean_sig / mean_sm:.2f}x higher than background.")
        
        else:
            # NF Evaluation (Log-Likelihood)
            sm_log_probs = model.log_prob(sm_tensor)
            mean_sm = torch.mean(sm_log_probs).item()

            sig_log_probs = model.log_prob(signal_tensor)
            mean_sig = torch.mean(sig_log_probs).item()

            print(f"\n[NF Sanity Check] Mean SM Test Log-Likelihood: {mean_sm:.4f}")
            print(f"[NF Signal Test]  Mean Stop Signal Log-Likelihood: {mean_sig:.4f}")
            print(f"-> Interpretation: Signal log-likelihood is lower (more anomalous) than background.")

if __name__ == "__main__":
    main()
