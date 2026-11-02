# Environment and Data Setup

Everyone on the team must use the same package versions and the same data split.

## Requirements
- Python (same major.minor as the team lead; check with `python --version`)
- Git
- About 3 GB free disk space (dataset cache is about 1.1 GB)

## Option A: one command (Windows PowerShell)
    git clone https://github.com/MohamedElHadary05/DL_Project.git
    cd DL_Project
    powershell -ExecutionPolicy Bypass -File setup.ps1

## Option B: manual steps
    git clone https://github.com/MohamedElHadary05/DL_Project.git
    cd DL_Project
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python download_data.py
    python -m src.check_env

If PowerShell blocks activation, run this first:
    Set-ExecutionPolicy -Scope Process Bypass

## Dataset
The dataset (CUB-200-2011 with captions) is downloaded automatically from
Hugging Face by `download_data.py` and cached in `data/hf_cache`.
It is not stored in the repo. Optional, to avoid rate limits:
    $env:HF_TOKEN="your_token"

## Verify
`python -m src.check_env` must print:
    classes     200
    train/val   4795 1199
    test        5793
    ENV OK
Also compare the `val split fingerprint` line with your teammates. It must be identical.

## Optional: NVIDIA GPU
Check with `nvidia-smi`. If a GPU exists, install the CUDA build of torch from
pytorch.org using the SAME torch version as in requirements.txt, then rerun
`python -m src.check_env` and confirm the `cuda` line says True.

## Git workflow
    git checkout -b your-name
    git add .
    git commit -m "describe your change"
    git push -u origin your-name
Commit from your own GitHub account. Never commit data/, mlruns/ or model files.