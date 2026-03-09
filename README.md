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
├── README.md
├── RTF+ALM.ipynb                    # REaLTabFormer + ALM (training + generation)
├── CTAB GAN+ ALM.ipynb              # CTAB-GAN+ + ALM (training + generation)
├── eval/                            # Evaluation scripts (privacy, utility, fidelity)
│   ├── evaluate_all.py              # Main entrypoint to reproduce paper evaluations
│   ├── commercial_tools_scores.py   # Evaluation of commercial synthetic data tools
│   ├── statistical_fidelity_analysis.py
│   ├── ml_utility_evaluation.py
│   └── adverserial_attacks.py
└── data/
    └── CTAB-GAN-Plus-outputs/       # Example synthetic outputs from CTAB-GAN+ (CSV)
```

The evaluation scripts expect an `ALM_Paper/` folder under `eval/` containing the **original** and **synthetic** CSVs for all datasets used in the paper (e.g. `diabetes_health_indicators_original.csv`, `*_synthetic_alm.csv`, `*_synthetic_rtf.csv`, etc.).
If you are a reviewer, this folder is provided as part of the **supplementary material / data archive**; after downloading it, place it at:

```
PsyGenTAB/eval/ALM_Paper/
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

## ⚙️ Environment Setup

**Recommended Python version:** 3.9–3.11

All core dependencies for the evaluation scripts and notebooks are listed in `requirements.txt`.

1. **Create and activate a virtual environment**
   - On macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. (Optional, for GPU) Install a CUDA-enabled build of PyTorch following the instructions from the official PyTorch documentation.


## ▶️ Reproducing Evaluation Results (Command Line)

The main entrypoint to reproduce the paper’s evaluation tables and figures is `eval/evaluate_all.py`.

From the repository root:

```bash
cd PsyGenTAB
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# Ensure ALM_Paper/ with original + synthetic CSVs is present:
# PsyGenTAB/eval/ALM_Paper/...

python eval/evaluate_all.py
```

By default, this:

- Evaluates **11 datasets** (e.g. `diabetes_health_indicators`, `adult_census_income`, `breast_cancer`, `vn_banking`, `lung_cancer`, `obesity`, `hypothyroid`, `liver_disorders`, `heart_failure_clinical_records`, `pir_vision_office`).
- Compares **ALM** vs **RTF** synthetic data.
- Writes per-dataset JSON reports and an aggregated `evaluation_summary.json` under:

```
eval/ALM_Paper/evaluation_reports/
```

### Optional components and external tools

Some evaluation components depend on external libraries that may be slow or licensed differently:

- **SDMetrics reports** (via `sdmetrics` / `sdv`)
- **MostlyAI QA reports** (via `mostlyai`)
- **Privacy attack framework** (via `privacy_evaluation_framework`, if available)

You can:

- **Disable specific components**:
  ```bash
  python eval/evaluate_all.py --skip-components privacy_attacks sdmetrics mostlyai_qa
  ```
- **Run only a subset of datasets**:
  ```bash
  python eval/evaluate_all.py --datasets diabetes_health_indicators adult_census_income
  ```


## ▶️ Running Experiments from Notebooks

The two main notebooks are:

- `RTF+ALM.ipynb`
- `CTAB GAN+ ALM.ipynb`

Each notebook:

- Trains the underlying generator (REaLTabFormer or CTAB-GAN+).
- Generates synthetic datasets at different privacy–utility operating points.
- Saves synthetic CSVs expected by the evaluation scripts.

These notebooks may require **additional, model-specific dependencies** (e.g. REaLTabFormer and CTAB-GAN+ packages).
We recommend using the same virtual environment as above, and installing any extra packages indicated at the top of each notebook.

To run a notebook:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter
jupyter notebook
```

Then open the desired notebook and execute all cells.

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


