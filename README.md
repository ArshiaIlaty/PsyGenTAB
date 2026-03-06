# PSyGenTAB

### Privacy-Preserving Synthetic Tabular Data Generation via Constrained Optimization

PSyGenTAB is a **model-agnostic framework** for generating synthetic tabular data with **explicit and controllable privacy–utility trade-offs**.

The framework formulates synthetic data generation as a **constrained optimization problem** and enforces privacy using the **Augmented Lagrangian Method (ALM)**. It integrates multi-dimensional utility and privacy metrics and works with both transformer-based and GAN-based tabular generators.

---

## 🚀 Features

* Constrained optimization formulation for synthetic data generation
* Explicit privacy threshold enforcement
* Composite utility and privacy metrics
* Sampling-time ALM optimization
* Compatible with:

  * REaLTabFormer (Transformer-based)
  * CTAB-GAN+ (GAN-based)
* Model-agnostic design (no architectural modification required)

---

## 📂 Repository Structure

```
.
├── RTF_ALM.ipynb              # REaLTabFormer + ALM implementation
├── CTAB GAN+ ALM.ipynb        # CTAB-GAN+ + ALM implementation
└── README.md
```

---

## 🧠 Method Overview

We optimize the following constrained objective:

```
L = Q − λ * max(0, P_min − P) − (μ/2) * (P_min − P)^2
```

Where:

* `Q` = composite utility score
* `P` = privacy score
* `P_min` = required privacy threshold
* `λ, μ` = adaptive ALM multipliers

ALM outer iterations progressively enforce privacy constraints while preserving data fidelity.

---

## 📊 Datasets Used

Experiments were conducted on:

* Adult Census Income
* Breast Cancer Wisconsin (Diagnostic)
* Diabetes Health Indicators
* PIR Vision Office
* Vietnam Banking Transactions



---

## ⚙️ Installation

### Requirements

* Python 3.9+
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Optuna (for hyperparameter tuning)



## ▶️ Running Experiments

1. Open the desired notebook:

   * `RTF_ALM.ipynb`
   * `CTAB GAN+ ALM.ipynb`

2. Run:

   * Baseline model training
   * Synthetic data generation
   * Privacy & utility evaluation
   * ALM outer-loop optimization

Each notebook is self-contained and reproduces the results reported in the paper.

---

## 🔐 Privacy Note

This framework reduces memorization and re-identification risk through constrained optimization.
It does **not** provide formal differential privacy guarantees unless explicitly integrated with DP mechanisms.

---

## 📄 Citation

If you use this repository in academic work, please cite the associated paper (citation details will be added upon publication).

---

## 📬 Support

For questions, reproducibility issues, or suggestions, please open an issue in this repository.


