# Running the PSyGenTAB study

The pipeline is in `psygentab/` (data contract, metrics, generators, PSyGenTAB core)
and `scripts/` (job runner, aggregation, ablations, attacks). Every run writes to
`results/runs/<dataset>/<generator>/split<s>_seed<k>/` and is skipped once its
`DONE` marker exists, so lanes can be stopped, restarted or split across machines.

## Lanes

| Lane | Generators | Protocol |
|------|------------|----------|
| `cpu` | bayesian_network, arf, tvae, ctgan, tabddpm, pategan_eps{1,3,10} | 5 splits x 1 seed |
| `gpu` | rtf, ctabgan_plus, ctabgan_plus_dp_eps{1,3,10} | 5 splits x 3 seeds |

## Local (CPU lane)

    conda activate psygentab
    caffeinate -i python -m scripts.run_all --lane cpu --workers 2 --threads 2

## AWS (GPU lane)

1. Copy the repository, including `data/raw/` and `results/runs/` (from the CPU lane).
2. `bash scripts/setup_aws.sh`
3. Run the GPU lane (several workers can share one GPU):

       conda activate psygentab
       nohup python -m scripts.run_all --lane gpu --workers 3 --threads 4 > logs/lane_gpu.log 2>&1 &

4. Progress: `tail -f logs/status_gpu.log`; per-job logs are in `logs/jobs/`.

## After all runs

    python -m scripts.tapas_attack --dataset heart_failure --generator <gen>   # shadow-model MIA
    python -m scripts.ablations     # P_min sweep, ALM vs penalty/oracle, iterations, weights
    python -m scripts.aggregate     # mean/SD/95% CI, Friedman+Nemenyi, Wilcoxon+Holm, LaTeX
