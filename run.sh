#!/usr/bin/env bash
# One-shot environment setup for AdaptiveHIT.
#
# Creates the main `adaptivehit` conda env (meta_learner/ + data_adapter/)
# plus one env per base_model/* submodule, using the exact env names
# pipeline/*.sh already hardcodes (conplex, transformerCPI, DeepConv-DTI,
# drugban) so those scripts work unmodified afterwards.
#
# Each base_model/* submodule vendors its own, independently pinned
# dependency stack from its original 2019-2023 publication (see each
# submodule's own README.md). Only ConPLex_dev ships a real environment.yml;
# TransformerCPI/DeepConv-DTI/DrugBAN don't, so their install commands below
# are reproduced verbatim from their README.md's documented requirements,
# not invented. Some pin old CUDA/TensorFlow versions that may not resolve
# on newer GPU drivers or conda solvers -- if one step below fails, that is
# an upstream base-model dependency problem, not this script; edit the
# relevant block by hand for your own CUDA setup and re-run (steps are
# idempotent and skip envs that already exist).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

conda_env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

if [ ! -f base_model/ConPLex_dev/README.md ]; then
    echo "[submodules] initializing git submodules"
    git submodule update --init --recursive
fi

# ---------------------------------------------------------------------------
# Apply AdaptiveHIT's integration patches to each pristine base_model/*
# submodule. Each submodule is pinned at the original authors' own upstream
# commit (not a fork) -- patches/<name>.patch (generated from the same diff
# documented in NOTICE.md) adds the main_integ_*.py/predict_*.py entrypoints
# and the few small upstream fixes on top. Idempotent: skips a submodule
# whose patch marker file already exists.
# ---------------------------------------------------------------------------
declare -A PATCH_MARKER=(
    [ConPLex_dev]=main_integ_conplex.py
    [DrugBAN]=main_integ_drugban.py
    [TransformerCPI]=GPCR/main_integ_trans.py
    [DeepConv-DTI]=main_integ_DeepConvDTI.py
)
for name in ConPLex_dev DrugBAN TransformerCPI "DeepConv-DTI"; do
    if [ -f "base_model/$name/${PATCH_MARKER[$name]}" ]; then
        echo "[patch:$name] already applied, skipping"
    else
        echo "[patch:$name] applying patches/$name.patch"
        git -C "base_model/$name" apply "$SCRIPT_DIR/patches/$name.patch"
    fi
done

# ---------------------------------------------------------------------------
# Main environment (meta_learner/ + data_adapter/) -- see environment.yml
# ---------------------------------------------------------------------------
if conda_env_exists adaptivehit; then
    echo "[adaptivehit] already exists, skipping"
else
    echo "[adaptivehit] creating from environment.yml"
    conda env create -f environment.yml
fi

# ---------------------------------------------------------------------------
# ConPLex (pipeline/*.sh expects env name: conplex)
# base_model/ConPLex_dev/environment.yml is a full `conda env export` from
# the original authors' machine (name: dsplat, python 3.7, cudatoolkit 10.2,
# hardcoded build strings) -- forced to the `conplex` name below so it lines
# up with pipeline/*.sh. Full-build-string exports like this are brittle
# across conda versions/channels/OSes; if it fails to solve, re-create it
# from base_model/ConPLex_dev/requirements.txt instead.
# ---------------------------------------------------------------------------
if conda_env_exists conplex; then
    echo "[conplex] already exists, skipping"
else
    echo "[conplex] creating from base_model/ConPLex_dev/environment.yml"
    conda env create -f base_model/ConPLex_dev/environment.yml -n conplex
fi

# ---------------------------------------------------------------------------
# TransformerCPI (pipeline/*.sh expects env name: transformerCPI)
# No environment.yml/requirements.txt shipped -- dependency versions below
# (python 3.6, pytorch>=1.2.0, rdkit==2019.03.3.0, gensim>=3.4.0) are exactly
# what base_model/TransformerCPI/README.md documents.
# ---------------------------------------------------------------------------
if conda_env_exists transformerCPI; then
    echo "[transformerCPI] already exists, skipping"
else
    echo "[transformerCPI] creating (python 3.6, pytorch>=1.2.0, rdkit==2019.03.3.0)"
    conda create -y -n transformerCPI python=3.6
    conda run -n transformerCPI conda install -y -c conda-forge rdkit=2019.03.3.0
    conda run -n transformerCPI pip install "torch>=1.2.0" numpy pandas "gensim>=3.4.0"
fi

# ---------------------------------------------------------------------------
# DeepConv-DTI (pipeline/*.sh expects env name: DeepConv-DTI)
# No environment.yml/requirements.txt shipped -- dependency ranges below
# ("tensorflow>1.0,<2.0", "keras>2.0") are exactly what
# base_model/DeepConv-DTI/README.md documents. python=3.7 chosen since
# TensorFlow 1.x has no wheels for newer Python; adjust if yours differs.
# ---------------------------------------------------------------------------
if conda_env_exists DeepConv-DTI; then
    echo "[DeepConv-DTI] already exists, skipping"
else
    echo "[DeepConv-DTI] creating (python 3.7, tensorflow>1.0,<2.0, keras>2.0)"
    conda create -y -n DeepConv-DTI python=3.7
    conda run -n DeepConv-DTI pip install "tensorflow>1.0,<2.0" "keras>2.0" numpy pandas scikit-learn
fi

# ---------------------------------------------------------------------------
# DrugBAN (pipeline/*.sh expects env name: drugban)
# Install sequence below is reproduced verbatim from the "Installation
# Guide" in base_model/DrugBAN/README.md (comet-ml is marked optional there
# and skipped here).
# ---------------------------------------------------------------------------
if conda_env_exists drugban; then
    echo "[drugban] already exists, skipping"
else
    echo "[drugban] creating (python 3.8, torch==1.7.1, cudatoolkit=10.2)"
    conda create -y -n drugban python=3.8
    conda run -n drugban conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
    conda run -n drugban conda install -y -c dglteam dgl-cuda10.2==0.7.1
    conda run -n drugban conda install -y -c conda-forge rdkit==2021.03.2
    conda run -n drugban pip install dgllife==0.2.8 scikit-learn yacs prettytable
fi

echo
echo "Done. 5 conda envs ready: adaptivehit, conplex, transformerCPI, DeepConv-DTI, drugban."
echo "Activate the one you need, e.g.: conda activate adaptivehit"
