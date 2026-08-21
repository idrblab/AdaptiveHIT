# Third-Party Notices

AdaptiveHIT's own code (`scripts/`, `bashes/`, `_ForFeatures/` and the
top-level `*.sh` drivers) is released under the MIT license (see `LICENSE`). The four base compound-protein
interaction (CPI) models it ensembles are vendored under `base_model/` --
copied as regular tracked files (not a git submodule/fork) from a specific
commit of the original authors' own repository, listed per-model below, with
AdaptiveHIT's own integration changes applied directly on top. Each vendored
model retains its own upstream license, listed below -- none of them are
relicensed by this project.

## ConPLex (`base_model/ConPLex_dev`)

- Upstream: https://github.com/samsledje/ConPLex_dev
- Vendored at commit: `6e7e27748092be86e0401206b90c7630f0d83c96`
- License: MIT
- Reference: Singh, R. et al. "Adapting protein language models for rapid
  DTI prediction." *PNAS* (2023). https://www.pnas.org/doi/10.1073/pnas.2220778120
- Changes made for AdaptiveHIT: added `main_integ_conplex.py` (training
  entrypoint), `predict_conplex.py`/`predict_conplex_em.py` (standalone
  prediction), and `prepare_data_test`/`setup_test`/`test_dataloader_test`
  methods plus a `drug_target_collate_fn_test` (zero-pads variable-length
  target embeddings, e.g. FoldSeek structure embeddings) on `src/data.py`'s
  `DTIDataModule` to support prediction on an arbitrary test set.
  `DTIDataModule` also gained `train_path`/`val_path`/`test_path`
  constructor parameters; their defaults are kept *relative*
  (`"train/test.csv"` etc.) to match the `<split>/test.csv` layout that
  `scripts/data_processed/process_data_base_model.py` emits, and so that
  `data_dir / <path>` resolves correctly whether `data_dir` is absolute or
  relative. The ProtBert loader in `src/featurizers/protein.py` prefers the
  local copy under `models/huggingface/transformers/Rostlabprot_bert`
  (README's `huggingface.tar`) and falls back to the `Rostlab/prot_bert`
  HuggingFace id when it isn't present, replacing a hardcoded
  training-cluster absolute path.

## DrugBAN (`base_model/DrugBAN`)

- Upstream: https://github.com/peizhenbai/DrugBAN
- Vendored at commit: `9923f8c99959e00263103ff9ac61ba0eaccc8e02`
- License: MIT
- Reference: Bai, P., Miljković, F., John, B., Lu, H. "Interpretable
  bilinear attention network with domain adaptation improves drug-target
  prediction." *Nature Machine Intelligence* (2023).
  https://doi.org/10.1038/s42256-022-00605-1
- Changes made for AdaptiveHIT: added `main_integ_drugban.py`,
  `predict_drugban.py`/`predict_drugban_em.py`, extended `trainer.py` with
  per-epoch val+test evaluation/logging and a new `Predicter` class for
  standalone inference; fixed a crash (also present upstream, unfixed)
  where molecules with more heavy atoms than `DRUG.MAX_NODES` (290) made
  `dataloader.py`'s virtual-node padding go negative and raise inside
  `torch.zeros` -- oversized molecules are now dropped with a warning
  before graph featurization (`utils.py`'s `drop_oversized_molecules`,
  called from all 3 entrypoints) instead of crashing, and
  `graph_collate_func` skips any that still reach it as a second layer of
  defense. `trainer.py`'s `save_result_best` writes the best checkpoint to
  both `<--output-model>` and the `*_best.model` name that `adaptivehit_train.sh`
  consumes.

## TransformerCPI (`base_model/TransformerCPI`)

- Upstream: https://github.com/lifanchen-simm/TransformerCPI
- Vendored at commit: `1726d17067047d4e0645d853c39c5b1fd7abdd09`
- License: Apache-2.0
- Reference: Chen, L. et al. "TransformerCPI: improving compound-protein
  interaction prediction by sequence-based deep learning with self-attention
  mechanism and label reversal experiments." *Bioinformatics* 36(16),
  4406-4414 (2020). https://doi.org/10.1093/bioinformatics/btaa524
- Changes made for AdaptiveHIT: added `GPCR/main_integ_trans.py`,
  `GPCR/predict_trans.py`/`predict_trans_em.py`,
  `GPCR/mol_featurizer_integ_retrain.py`; fixed a metric-computation bug in
  `GPCR/model.py` (precision_recall_curve output was mislabeled as tpr/fpr)
  and modernized `GPCR/Radam.py`'s deprecated PyTorch API calls;
  `main_integ_trans.py`/`predict_trans.py`/`predict_trans_em.py` use a
  batched, GPU-optimized data pipeline (`OptimizedDTIDataset`, a padded
  collate function, in-memory feature caching, explicit CUDA memory
  cleanup) in place of upstream's per-sample loop; the retraining split is
  named `val` consistently across `mol_featurizer_integ_retrain.py`,
  `main_integ_trans.py` and `base_train.sh` (an earlier `dev`/`val`
  mismatch made the featurizer write `TransformerCPI/data/dev/` while
  training read `data/val/`); replaced hardcoded training-cluster absolute
  paths to `word2vec_30.model` with paths relative to the script's own
  location.

## DeepConv-DTI (`base_model/DeepConv-DTI`)

- Upstream: https://github.com/GIST-CSBL/DeepConv-DTI
- Vendored at commit: `527d0fa049caed96b7fcab09151f870c025dddb2`
- **License: GPL-3.0** -- vendored as its own directory and invoked as an
  independent process via a separate conda env (never imported or
  statically linked into AdaptiveHIT's own code), so AdaptiveHIT's own
  MIT-licensed code is unaffected; the GPL-3.0 terms apply to
  `base_model/DeepConv-DTI` itself.
- Changes made for AdaptiveHIT: added `main_integ_DeepConvDTI.py` (adds an
  HDF5-based streaming data pipeline for datasets too large to fit in
  memory), `predict_deepconv.py`/`predict_deepconv_em.py`.

## Third-party pretrained representations

None of the three below are bundled in this repository; each is fetched by
`pull_external_assets.sh` or downloaded manually per the **Manual Resource
Download** table in `README.md`, and all three are `.gitignore`d.

**X-Mol** (`_ForFeatures/xmol/`, ~1GB of weights + a ~1.35GB bundled Python
runtime): a frozen, pretrained SMILES/molecule representation model
(ERNIE-style transformer, 768-dim, PaddlePaddle checkpoint format) used as
the compound feature extractor. No paper citation or redistribution license
for these specific weights was available at release time -- if you can trace
their original source/license, please update this section before archiving.

**ProtBert** (`base_model/ConPLex_dev/models/huggingface/`, ~1.56GB):
Rostlab's `prot_bert`, ConPLex's protein featurizer. Downloaded from the
HuggingFace Hub automatically when the local copy is absent.

**ESM-2** (`esm2_t36_3B_UR50D`, ~5.4GB): Meta AI's protein language model
(`fair-esm`, MIT-licensed), AdaptiveHIT's protein representation.
Reference: Lin, Z. et al. "Evolutionary-scale prediction of atomic-level
protein structure with a language model." *Science* 379(6637), 1123-1130
(2023). https://doi.org/10.1126/science.ade2574

## Data provenance

`dataset/toy_dataset/{train,val,test}.csv` is shipped as a small,
ready-to-run worked example of the full pipeline.

The datasets behind the manuscript's benchmarks and robustness experiments
-- `dataset/Multisimi/`, `dataset/ChEMBL/`, `dataset/BindingDB/`,
`dataset/HUMAN/`, `dataset/HUMAN_cold_pair/` -- are **not tracked in this
repository**. Download `dataset.tar` via `pull_external_assets.sh` or the
link in `README.md`'s **Manual Resource Download** table and extract it at
the repository root so the splits land under `dataset/`.

These splits derive from a raw ChEMBL-derived activity table (Molecule
ChEMBL ID, SMILES, Target ChEMBL ID, UniProt accession, document year,
label, standard type/relation/value/units, gene names, organism, sequence).
That raw table is not included, and no script recovering the exact transform
from it to the shipped splits was available at release time.
