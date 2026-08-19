# Third-Party Notices

AdaptiveHIT's own code (`meta_learner/`, `data_adapter/`, `pipeline/`) is
released under the MIT license (see `LICENSE`). The four base compound-protein
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
  `DTIDataModule` to support prediction on an arbitrary test set; replaced
  a hardcoded training-cluster absolute path in `src/featurizers/protein.py`'s
  ProtBert loader with the portable `Rostlab/prot_bert` HuggingFace model ID.
  `DTIDataModule`'s `train_path`/`val_path`/`test_path` constructor
  parameters (added alongside the above) briefly regressed to the wrong
  default filenames (`"train/test.csv"` etc. instead of `"train.csv"`),
  which broke `main_integ_conplex.py` (it relies on the defaults); fixed to
  match the real flat `<data_dir>/{train,val,test}.csv` layout used
  throughout `dataset/`.

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
  defense.

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
  cleanup) in place of upstream's per-sample loop; fixed a val/dev split
  naming mismatch in `mol_featurizer_integ_retrain.py`'s non-random-split
  branch; replaced a hardcoded training-cluster absolute path to
  `word2vec_30.model` with a path relative to the script's own location.

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

**X-Mol** (`data_adapter/xmol_weights/`, ~327MB, bundled via Git LFS): a
frozen, pretrained SMILES/molecule representation model (ERNIE-style
transformer, 768-dim, PaddlePaddle checkpoint format) used here as a
compound feature extractor via `data_adapter/xmol_embed.py`, a from-scratch
PyTorch re-implementation of the inference path (no PaddlePaddle runtime
dependency). No paper citation or redistribution license for these specific
weights was available at release time -- if you can trace their original
source/license, please update this section before archiving.

**ESM-2** (`esm2_t36_3B_UR50D`, ~5.4GB, **not bundled** in this repository
or its Git LFS objects): Meta AI's protein language model, downloaded
on first use via the `fair-esm` package (MIT-licensed) from
`https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt` --
see the Quickstart section in `README.md` for the exact download command.
Reference: Lin, Z. et al. "Evolutionary-scale prediction of atomic-level
protein structure with a language model." *Science* 379(6637), 1123-1130
(2023). https://doi.org/10.1126/science.ade2574

## Data provenance

`Smiles_Sequence_label_year_gene_activalue.csv`, a 529MB raw ChEMBL-derived
table (Molecule ChEMBL ID, SMILES, Target ChEMBL ID, UniProt accession,
document year, label, standard type/relation/value/units, gene names,
organism, sequence), is the original source data this project's training
splits were derived from. It is **not included** in this repository, and no
script recovering the exact transform from that raw table to the
`data/toy_dataset/{train,val,test}.csv` splits shipped here was available at
release time -- those splits are shipped as-is as a small, ready-to-use
worked example of the full pipeline.

A larger 5-fold cross-validation variant of this dataset
(`multisimi_dataset/`, ~417MB) was used for the robustness experiments
described in the manuscript but is not included in this repository to keep
it lean; contact the corresponding author for access.
