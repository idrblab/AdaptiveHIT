#!/usr/bin/env bash
# One-shot environment setup for AdaptiveHIT.
#
# Creates the main `adaptivehit` conda env (meta_learner/ + data_adapter/)
# plus one env per base_model/* model, using the exact env names
# pipeline/*.sh already hardcodes (conplex, transformerCPI, DeepConv-DTI,
# drugban) so those scripts work unmodified afterwards.
#
# Each base_model/* directory is vendored (regular tracked files, not a git
# submodule) with its own independently pinned dependency stack from its
# original 2019-2023 publication (see each model's own README.md and
# NOTICE.md for upstream provenance + the modifications applied on top).
# Only ConPLex_dev ships a real environment.yml;
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

# ---------------------------------------------------------------------------
# Anaconda's `pkgs/main`/`pkgs/r` channels (aka `defaults`) require accepting
# their Terms of Service before conda will install anything from them --
# every `conda create -n X python=Y` below hits this, and it's where
# `cudatoolkit` actually lives even when installing "from" -c pytorch. In
# non-interactive mode (this script uses -y throughout) conda refuses to
# prompt and just errors out, which looks like every env below failing to
# install on a fresh machine that hasn't accepted these ToS yet. Accepting
# here once, non-interactively, is conda's own documented mechanism for
# scripted/CI installs.
# ---------------------------------------------------------------------------
echo "[tos] accepting Anaconda ToS for pkgs/main and pkgs/r (needed for cudatoolkit etc.)"
conda tos accept --override-channels -c https://repo.anaconda.com/pkgs/main -c https://repo.anaconda.com/pkgs/r

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
# hardcoded build strings pinned to openssl<3) -- confirmed by actually
# trying it that it no longer solves on a 2026 conda/channel snapshot.
# base_model/ConPLex_dev/requirements.txt is NOT a full substitute either --
# it's only 3 supplementary pip packages (dgl, pysmiles, mol2vec) on top of
# a base scientific/DL stack it doesn't declare at all.
#
# The recipe below was built from ConPLex_dev's actual imports (not the
# broken environment.yml) and verified end-to-end by actually running
# main_integ_conplex.py to a completed training epoch on real data. Every
# pin exists because leaving it unpinned broke something concretely:
#   - dscript unpinned resolves to 0.2.8, an unrelated single-cell-genomics
#     release that drags in torch 2.8 + a huge irrelevant dependency tree
#     (scanpy, tiledbsoma, cellxgene-census, ...). dscript==0.1.9 is what
#     the original environment.yml actually pinned.
#   - plain `pip install torch==1.11.0` resolves to the +cu102 build (CUDA
#     10.2), whose arch list tops out at sm_70 (Volta) -- no kernels for
#     Turing/Ampere/Hopper at all, so it crashes with "no kernel image is
#     available for execution on the device" on any GPU newer than that,
#     A100 included. `--extra-index-url .../whl/cu113` pins the +cu113
#     build instead (confirmed to exist for cp39), which does carry Ampere
#     kernels.
#   - dgl (latest) needs its C++ "graphbolt" extension built for the exact
#     torch minor version installed and needs torchdata.datapipes (removed
#     in modern torchdata) + pydantic transitively but undeclared. dgl==0.9.1
#     predates graphbolt entirely, sidestepping all of that. The old fix
#     (`pip install dgl==0.9.1 -f https://data.dgl.ai/wheels/torch-1.11/...`)
#     no longer works -- that custom wheel index URL now 403s (dgl.ai
#     restructured its hosting since), and pip silently falls back to the
#     plain PyPI `dgl` package instead of erroring, which is a small
#     (~5MB) CPU-only build -- so GPU was silently never used here at all,
#     independent of A100/Ampere. `-c dglteam dgl-cuda11.6==0.9.1` is a
#     real GPU build of the exact same dgl version, confirmed available.
#   - torchmetrics/pytorch_lightning unpinned pull torch>=2.0, silently
#     upgrading torch out from under dgl 0.9.1 again. 0.9.3/1.8.6 are the
#     newest versions of each still compatible with torch 1.11.
#   - transformers==4.18.0 (the environment.yml's own pin) can no longer
#     download models from huggingface.co in 2026: HF's server now returns
#     redirect responses with a relative (schemeless) Location header, and
#     4.18.0's hand-rolled downloader doesn't resolve it against the
#     request's base URL (`MissingSchema` error) -- a real client/server
#     compatibility decay, not an environment bug. transformers==4.30.2
#     (+ a huggingface_hub/tokenizers pair it's compatible with) handles
#     the redirect correctly via standard `requests` behavior.
#   - deepchem is imported by src/featurizers/molecule.py but isn't in
#     requirements.txt either; ==2.6.1 matches the original environment.yml.
# ---------------------------------------------------------------------------
if conda_env_exists conplex; then
    echo "[conplex] already exists, skipping"
else
    echo "[conplex] creating (python 3.9, torch==1.11.0+cu113, dgl-cuda11.6==0.9.1, transformers==4.30.2)"
    conda create -y -n conplex python=3.9
    conda run -n conplex conda install -y -c conda-forge rdkit=2022.03.2
    conda run -n conplex conda install -y -c dglteam dgl-cuda11.6==0.9.1
    conda run -n conplex pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    conda run -n conplex pip install \
        "dscript==0.1.9" "fair-esm==0.4.2" \
        "transformers==4.30.2" "huggingface_hub>=0.14,<0.17" "tokenizers>=0.13,<0.14" \
        "scikit-learn==1.0.2" "pandas==1.3.5" "numpy<2.0" omegaconf tqdm \
        "torchmetrics==0.9.3" "pytorch_lightning==1.8.6" h5py \
        pysmiles "mol2vec @ git+https://github.com/samoturk/mol2vec" pytdc \
        "deepchem==2.6.1"
fi

# ---------------------------------------------------------------------------
# TransformerCPI (pipeline/*.sh expects env name: transformerCPI)
# No environment.yml/requirements.txt shipped -- base version constraints
# (python 3.6, pytorch>=1.2.0, rdkit==2019.03.3.0, gensim>=3.4.0) are what
# base_model/TransformerCPI/README.md documents, but that list is incomplete
# in practice (verified by actually running the retraining pipeline):
# scikit-learn is imported by model.py but isn't listed anywhere upstream,
# and gensim>=3.4.0 unpinned resolves to gensim 4.x whose smart_open
# dependency uses `from __future__ import annotations` (Python 3.7+ syntax),
# a hard SyntaxError under this env's Python 3.6. gensim==3.8.3 +
# smart_open<5.0 is the newest combination that's actually importable here.
# `torch>=1.2.0` unpinned resolves to 1.10.2 (the newest torch with cp36
# wheels), but as the plain +cu102 build -- same Ampere-incompatible
# situation as ConPLex (see above), just landed on by version-resolution
# instead of an explicit pin. Pinning to the +cu113 build (confirmed to
# exist for cp36) fixes it the same way.
# ---------------------------------------------------------------------------
if conda_env_exists transformerCPI; then
    echo "[transformerCPI] already exists, skipping"
else
    echo "[transformerCPI] creating (python 3.6, torch==1.10.2+cu113, rdkit==2019.03.3.0)"
    conda create -y -n transformerCPI python=3.6
    conda run -n transformerCPI conda install -y -c conda-forge rdkit=2019.03.3.0
    conda run -n transformerCPI pip install "torch==1.10.2+cu113" --extra-index-url https://download.pytorch.org/whl/cu113 numpy pandas scikit-learn "gensim==3.8.3" "smart_open<5.0"
fi

# ---------------------------------------------------------------------------
# DeepConv-DTI (pipeline/*.sh expects env name: DeepConv-DTI)
# No environment.yml/requirements.txt shipped -- base dependency ranges
# ("tensorflow>1.0,<2.0", "keras>2.0") are what
# base_model/DeepConv-DTI/README.md documents, but that's incomplete in
# practice (verified by actually running the retraining pipeline):
# - "keras>2.0" unpinned resolves to a modern standalone Keras (2.11) built
#   against TF2's internal API (`tf.__internal__`), which doesn't exist in
#   TF1 -- AttributeError at import time. keras==2.3.1 is the last release
#   built for TF1. The code imports standalone `keras` (not `tensorflow.keras`)
#   throughout, so this stays pinned regardless of which tensorflow backend
#   below actually provides `import tensorflow`.
# - pip's default (modern) protobuf is incompatible with TF 1.15's
#   pre-compiled *_pb2.py files ("Descriptors cannot not be created
#   directly"). protobuf<3.20 fixes it (TF's own suggested workaround).
# - plain `pip install "tensorflow>1.0,<2.0"` resolves to tensorflow==1.15.5
#   from PyPI, which is CPU-only (confirmed: tf.test.is_built_with_cuda()
#   is False) -- TF1.x's PyPI "tensorflow" package was always CPU-only, GPU
#   was a separate "tensorflow-gpu" package. This doesn't crash (the code
#   already does tf.test.is_gpu_available() -> falls back to /cpu:0), it
#   just silently never uses the GPU. Official TF 1.15 GPU wheels also
#   predate Ampere, same problem class as the other 3 models. NVIDIA
#   maintains "nvidia-tensorflow", a patched TF 1.15 build with newer-GPU
#   (incl. Ampere) support, published on its own package index -- swapped
#   in below. NOT verified end-to-end from this sandbox (pypi.ngc.nvidia.com
#   wasn't reachable here to test); verify this block actually installs and
#   that `tf.test.is_built_with_cuda()` is True on your real GPU machine,
#   and adjust the nvidia-tensorflow version pin if it's no longer available.
# python=3.7 chosen since TensorFlow 1.x has no wheels for newer Python;
# adjust if yours differs.
# ---------------------------------------------------------------------------
if conda_env_exists DeepConv-DTI; then
    echo "[DeepConv-DTI] already exists, skipping"
else
    echo "[DeepConv-DTI] creating (python 3.7, nvidia-tensorflow==1.15.5, keras==2.3.1)"
    conda create -y -n DeepConv-DTI python=3.7
    conda run -n DeepConv-DTI pip install nvidia-pyindex
    conda run -n DeepConv-DTI pip install "nvidia-tensorflow==1.15.5+nv22.12" "keras==2.3.1" "protobuf<3.20" numpy pandas scikit-learn
fi

# ---------------------------------------------------------------------------
# DrugBAN (pipeline/*.sh expects env name: drugban)
# Install sequence is DrugBAN's own documented "Installation Guide" in
# base_model/DrugBAN/README.md (comet-ml is marked optional there and
# skipped here), except cudatoolkit is bumped 10.2 -> 11.0 (with a matching
# dgl-cuda11.0 build): CUDA 10.2 predates Ampere and has no kernels for
# sm_80 (A100) -- torch/dgl built against it install fine but fail at
# runtime on an A100 with "no kernel image is available for execution on
# the device". cudatoolkit=11.0 solves to real GPU builds
# (pytorch-1.7.1-cuda11.0.221, torchvision-0.8.2-cu110) and dgl-cuda11.0==0.7.1
# exists on the dglteam channel for the same dgl version, so this is a
# drop-in swap with no other version changes.
# ---------------------------------------------------------------------------
if conda_env_exists drugban; then
    echo "[drugban] already exists, skipping"
else
    echo "[drugban] creating (python 3.8, torch==1.7.1, cudatoolkit=11.0)"
    conda create -y -n drugban python=3.8
    conda run -n drugban conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
    conda run -n drugban conda install -y -c dglteam dgl-cuda11.0==0.7.1
    conda run -n drugban conda install -y -c conda-forge rdkit==2021.03.2
    conda run -n drugban pip install dgllife==0.2.8 scikit-learn yacs prettytable
fi

echo
echo "Done. 5 conda envs ready: adaptivehit, conplex, transformerCPI, DeepConv-DTI, drugban."
echo "Activate the one you need, e.g.: conda activate adaptivehit"
