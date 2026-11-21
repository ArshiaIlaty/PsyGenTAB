#!/usr/bin/env bash
# One-time setup of the PSyGenTAB study environment on a CUDA (AWS) instance.
# Usage, from the repository root:  bash scripts/setup_aws.sh
set -euo pipefail

ENV=psygentab
CTAB_COMMIT=6a6f901   # Team-TUD/CTAB-GAN-Plus commit used for all runs

if ! command -v conda >/dev/null; then
  echo "conda not found: install Miniconda first" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV" python=3.10
conda activate "$ENV"
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-study.txt

mkdir -p third_party
if [ ! -d third_party/CTAB-GAN-Plus ]; then
  git clone https://github.com/Team-TUD/CTAB-GAN-Plus.git third_party/CTAB-GAN-Plus
fi
git -C third_party/CTAB-GAN-Plus checkout -q "$CTAB_COMMIT"

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible to torch"
print("CUDA OK:", torch.cuda.get_device_name(0))
from psygentab.datasets import describe_all
print(describe_all().to_string(index=False))
EOF
