# Anomaly Detection in Experimental High Energy Physics (EHEP)

An open-source, modular framework for unsupervised anomaly detection in high-energy collider physics. This repository implements a dual-paradigm pipeline—combining **Variational Autoencoders (VAEs)** and **RealNVP Normalizing Flows**—to identify new physics signals (such as scalar top quarks) in unbinned event-level kinematic phase spaces.

## 📌 Project Overview
Unsupervised anomaly detection trains exclusively on Standard Model (SM) background events to learn nominal kinematic distributions. New physics phenomena are subsequently identified as outliers without requiring prior supervision or simulated signal training data.

### Current Status: Phase 1 Complete 
* **Dual-Model Support:** Unified training and evaluation architecture supporting both VAE reconstruction error and Normalizing Flow exact log-likelihood ($\log p(x)$) estimation.
* **Kinematic Feature Engineering:** Operates on unbinned flat ROOT trees, focusing on key variables: missing transverse energy ($\log\text{MET}$), lepton transverse momentum ($\log p_T^{\text{lep}}$), and transverse mass ($m_T$).
* **Decoupled Data Pipeline:** Standardized parsing, luminosity weighting, and static train/test splitting.

---

## 🛠️ System Architecture & Modularity

The framework is decoupled into data preprocessing, generalized model definitions, and unified execution scripts:
1. **`prepare_tree_data.py` & `prepare_hist_data.py`**: Parsers for raw ROOT trees and histogram-format data structures, handling log-transformations, luminosity weighting, and static dataset splitting.
2. **`VAEmodel.py` & `NFmodel.py`**: Model definitions for Dense VAEs and generalized even/odd RealNVP Normalizing Flows.
3. **`train.py` & `evaluate.py`**: Unified command-line interface scripts controlled via `--model vae` or `--model nf` runtime arguments.

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone [https://github.com/kmandal/AnomalyDetection-EHEP.git](https://github.com/kmandal/AnomalyDetection-EHEP.git)
cd AnomalyDetection-EHEP
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Data Preparation

Place your raw Standard Model and signal ROOT files inside the `data/` directory, then generate the processed feature splits using your preferred format:

* **For Event-Level Tree Data:**
```bash
python src/prepare_tree_data.py
```
* **For Histogram-Based Data:**
```bash
python src/prepare_hist_data.py
```

### 3. Execution via Unified Flags

**Option A: Variational Autoencoder (VAE) Pipeline**
* **Train on SM Background:**
```bash
python src/train.py --model vae
```
* **Evaluate Anomaly Separation:**
```bash
python src/evaluate.py --model vae
```

**Option B: Normalizing Flow (RealNVP) Pipeline**
* **Train Density Estimator:**
```bash
python src/train.py --model nf
```
* **Evaluate Exact Log-Likelihood:**
```bash
python src/evaluate.py --model nf
```


## 🗺️ Roadmap: Phase 2 and Beyond
*[ ] **Quantile Regression Neural Networks:** Implementing direct conditional quantile bound estimation to isolate phase-space boundary outliers rigorously.

*[ ] **Extended Feature Spaces:** Expanding dimensionality beyond 3 kinematic variables to include angular correlations and multi-jet topologies.


## 🤝 Contributing
Contributions, issue reports, and alternative flow architectures are welcome! Feel free to open an issue or submit a pull request.
