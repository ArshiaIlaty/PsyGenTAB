# PSyGenTAB

### Privacy-Constrained Synthetic Health Data Generation at Sampling Time

**Website and interactive results:** https://arshiailaty.github.io/PsyGenTAB/

PSyGenTAB turns the privacy of a synthetic-data release into a floor that the data custodian sets, $P \geq P_{\min}$. It enforces that floor at **sampling time**, for any already-trained tabular generator and without retraining.

How it works:

1. From the trained generator, PSyGenTAB draws candidate releases. A candidate is a sampling configuration (for example, a temperature) plus a record-acceptance threshold.
2. Every candidate is scored for utility $U$ and privacy $P$ on a calibration holdout that the generator never saw.
3. The release is the most useful candidate that meets the floor. This choice is solved exactly with an augmented Lagrangian.

Every reported number is recomputed on a separate, untouched test split, and every release is checked with five attack families.

This repository contains the full study pipeline:

- seven generators plus differentially private (DP) baselines
- ten datasets
- 1,600 runs and 3,250 evaluated releases
- the attacks, ablations and statistics
- the code that builds the paper's tables and the website

> The original notebooks (`RTF+ALM.ipynb`, `eval/`) are kept for reference. The study code is the `psygentab/` package together with `scripts/`.

---

## Repository structure

```
psygentab/              library
  datasets.py           dataset registry, cleaning, 60/20/20 split contract
  encoding.py           shared numeric encoding for every distance-based metric
  generators.py         uniform fit/sample interface: REaLTabFormer, CTAB-GAN+, synthcity zoo
  ctabgan_dp.py         DP-SGD CTAB-GAN+ (weight-clipping and gradient-penalty critics, RDP accounting)
  core.py               candidate scoring, PSyGenTAB selectors (ALM, minimal intervention, baselines)
  metrics.py            utility, fidelity, privacy, attacks, fairness and diversity metrics
  evaluate.py           full evaluation of one released dataset on the test split
scripts/
  run_one.py, run_all.py     one run / the whole grid (CPU and GPU lanes)
  tapas_attack.py            shadow-model membership inference (TAPAS-style)
  ablations.py               P_min sweep, ALM vs. penalty/oracle, outer iterations, weights
  aggregate.py, leaderboard.py, ci.py   tables, ranks, Friedman/Nemenyi, Wilcoxon+Holm, 95% CIs
  figures.py, paper_tables.py, supplement.py   paper figures, tables and supplementary material
  build_site.py, site_template.html            the project website
  rescore_holdout_size.py    recomputes test-side proximity metrics after the equal-reference-size fix
third_party/CTAB-GAN-Plus    official CTAB-GAN+ (git submodule, unmodified)
RUN_STUDY.md                 how to reproduce the study (local CPU lane + AWS GPU lane)
requirements-study.txt       pinned environment
```

## Setup

```bash
git clone --recurse-submodules https://github.com/ArshiaIlaty/PsyGenTAB.git
cd PsyGenTAB
conda create -n psygentab python=3.10 -y && conda activate psygentab
pip install -r requirements-study.txt
```

## Data

The datasets are public and are not redistributed here. Place each one as `data/raw/<name>.csv`, or point the `PSY_DATA` environment variable at another folder. The expected names and target columns are:

| name | target | domain |
|---|---|---|
| `breast_cancer` (WDBC, 569 rows) | `diagnosis` | clinical |
| `diabetes` | `Diabetes_012` | health survey |
| `heart_failure` | `DEATH_EVENT` | clinical |
| `hypothyroid` | `binaryClass` | clinical |
| `liver_disorders` | `drinks > 3` (derived; `selector` is dropped) | clinical |
| `obesity` | `NObeyesdad` | health survey |
| `parkinsons` | `status` | clinical |
| `adult` | `Target` | demographics |
| `pir_vision` | `Target` | sensor |
| `vn_banking` | `is_churned` | financial |

Cleaning, column types and the 60/20/20 train / calibration / test split (5 splits) are defined in `psygentab/datasets.py`.

## Reproducing the study

See [RUN_STUDY.md](RUN_STUDY.md). In short:

```bash
python -m scripts.run_all --lane cpu      # BayesNet, ARF, TVAE, CTGAN, TabDDPM, PATE-GAN
python -m scripts.run_all --lane gpu      # REaLTabFormer, CTAB-GAN+, DP-CTAB-GAN+ (both variants)
python -m scripts.tapas_attack --dataset heart_failure --generator rtf
python -m scripts.ablations && python -m scripts.aggregate && python -m scripts.figures
python -m scripts.build_site              # website -> site/
```

Runs are resumable: finished runs leave a `DONE` marker and are skipped.

## Citation

```bibtex
@article{ilaty2026psygentab,
  title   = {PSyGenTAB: Privacy-Constrained Synthetic Health Data Generation at Sampling Time},
  author  = {Ilaty, Arshia and Shirazi, Hossein and Chitale, Manasi and Hegde, Kedar and Ramesh, Dhanalakshmi
             and Manjunath, Rashmi S. and Rahmani, Amir M. and Homayouni, Hajar},
  journal = {under review},
  year    = {2026}
}
```
