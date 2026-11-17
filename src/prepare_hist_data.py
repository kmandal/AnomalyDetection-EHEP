import numpy as np
import pandas as pd
import uproot
import os

def sample_from_th1f(histogram, n_samples=50000):
    """
    Takes an uproot TH1F object, extracts bin edges and contents,
    and samples unbinned points weighted by bin height.
    """
    counts, edges = histogram.to_numpy()
    # Normalize counts to form a probability distribution
    total = np.sum(counts)
    if total <= 0:
        raise ValueError("Histogram is empty!")
    probs = counts / total
    # Choose bin indices based on bin heights
    bin_indices = np.random.choice(len(counts), size=n_samples, p=probs)
    # Add uniform random jitter within each bin width to make it continuous
    bin_widths = np.diff(edges)
    sampled_values = edges[bin_indices] + np.random.uniform(0, bin_widths[bin_indices], size=n_samples)
    print(len(bin_indices), ' bin_widths: ', bin_widths, ' len(bin_widths): ', len(bin_widths), ' len(sampled_values): ', len(sampled_values))
    return sampled_values

def process_sm_histograms(root_files_dict, variable_name, output_csv):
    """
    Combines histograms from multiple SM processes and saves to CSV.
    root_files_dict: {"TTbar": "path/to/ttbar.root", "Wjets": "path/to/wjets.root"}
    """
    all_sampled_values = []
    
    for process_name, filepath in root_files_dict.items():
        print(f"Reading {variable_name} for process {process_name} from {filepath}...")
        with uproot.open(filepath) as f:
            # Assuming histogram name format like MET_TTbar or LeppT_Wjets
            hist_key = f"{variable_name}_{process_name}"
            if hist_key not in f:
                # Fallback if naming convention differs
                keys = f.keys()
                hist_key = [k for k in keys if variable_name in k][0]
            
            hist = f[hist_key]
            
            # Sample events from this process proportional to its luminosity-scaled content
            # (If you have multiple files, n_samples can be scaled by process cross-section/lumi)
            n_samples_proc = 30000 
            vals = sample_from_th1f(hist, n_samples=n_samples_proc)
            all_sampled_values.append(vals)
            
    # Combine all SM processes together
    combined_values = np.concatenate(all_sampled_values)
    
    # Apply log transform if checking heavy tails (like MET or pT)
    # For single variable like MET, input_dim = 1
    transformed_vals = np.log(combined_values + 1e-5)
    
    df = pd.DataFrame(transformed_vals, columns=[f"log_{variable_name}"])
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Successfully created combined SM dataset at {output_csv} with {len(df)} events.")

if __name__ == "__main__":
    # Example dictionary of your SM ROOT files
    sm_files = {
        "TTbar": "data/VarHist_TTbar.root"
        # You can add other SM files here when ready:
        # "Wjets": "VarHist_Wjets.root"
    }
    
    # Test converting the MET histogram
    process_sm_histograms(
        root_files_dict=sm_files,
        variable_name="MET",
        output_csv="data/sm_TTbar_hist.csv"
    )
