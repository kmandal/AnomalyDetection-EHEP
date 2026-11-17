from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import uproot
import os

def process_and_split_trees(process_files: dict, tree_name: str):
    all_dfs = []
    
    for process_name, file_path in process_files.items():
        print(f"Opening {process_name} from {file_path}...")
        if not os.path.exists(file_path):
            continue
            
        with uproot.open(file_path) as f:
            tree = f[tree_name]
            met = tree["MET"].array(library="np")
            pt_lep = tree["LeppT"].array(library="np")
            mt = tree["LepMT"].array(library="np")
            weights = tree["LumiWeight"].array(library="np")
            
        log_met = np.log(met + 1e-5)
        log_pt_lep = np.log(pt_lep + 1e-5)
        
        df_proc = pd.DataFrame({
            "log_MET": log_met,
            "log_pT_lep": log_pt_lep,
            "mT": mt,
            "weight": weights,
            "process": process_name
        })
        all_dfs.append(df_proc)
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Perform a strict 80/20 split with a fixed random seed for reproducibility
    df_train, df_test = train_test_split(combined_df, test_size=0.2, random_state=42)
    
    os.makedirs("data", exist_ok=True)
    df_train.to_csv("data/sm_train.csv", index=False)
    df_test.to_csv("data/sm_test.csv", index=False)
    
    print(f"\nSuccessfully split dataset!")
    print(f"Training events saved to data/sm_train.csv: {len(df_train)}")
    print(f"Test events saved to data/sm_test.csv: {len(df_test)}")

if __name__ == "__main__":
    sm_root_files = {
        "TTbar": "data/VarTree_TTbar.root",
        "QCD": "data/VarTree_QCD.root",
        "WJets": "data/VarTree_WJetsToLNu.root",
        "ZJets": "data/VarTree_ZJetsToNuNu.root",
        "ST": "data/VarTree_ST.root",
        "VV": "data/VarTree_VV.root",
        "TTV": "data/VarTree_TTV.root"
    }
    process_and_split_trees(sm_root_files, "Events")
