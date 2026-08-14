# Third-Party Notices

AdaptiveHIT's own code (`meta_learner/`, `data_adapter/`, `pipeline/`) is
released under the MIT license (see `LICENSE`). The four base compound-protein
interaction (CPI) models it ensembles are each a pristine, unmodified git
submodule under `base_model/`, pinned directly at a commit from the original
authors' own repository -- not a maintained fork. The minimal
additions/patches needed to plug each one into this project's training and
prediction pipeline live in `patches/<name>.patch` in this repository (`git
apply`'d onto the submodule by `run.sh`); see that patch file for the exact
diff against upstream. Each submodule retains its own upstream license,
listed below -- none of them are relicensed by this project.

## ConPLex (`base_model/ConPLex_dev`)

- Upstream: https://github.com/samsledje/ConPLex_dev
- License: MIT
- Reference: Singh, R. et al. "Adapting protein language models for rapid
  DTI prediction." *PNAS* (2023). https://www.pnas.org/doi/10.1073/pnas.2220778120
- Changes made for AdaptiveHIT: added `main_integ_conplex.py` (training
  entrypoint), `predict_conplex.py`/`predict_conplex_em.py` (standalone
  prediction), and `prepare_data_test`/`setup_test`/`test_dataloader_test`
  methods on `src/data.py`'s `DTIDataModule` to support prediction on an
  arbitrary test set.

## DrugBAN (`base_model/DrugBAN`)

- Upstream: https://github.com/peizhenbai/DrugBAN
- License: MIT
- Reference: Bai, P., Miljković, F., John, B., Lu, H. "Interpretable
  bilinear attention network with domain adaptation improves drug-target
  prediction." *Nature Machine Intelligence* (2023).
  https://doi.org/10.1038/s42256-022-00605-1
- Changes made for AdaptiveHIT: added `main_integ_drugban.py`,
  `predict_drugban.py`/`predict_drugban_em.py`, extended `trainer.py` with
  per-epoch val+test evaluation/logging and a new `Predicter` class for
  standalone inference, minor validation additions to `dataloader.py`.

## TransformerCPI (`base_model/TransformerCPI`)

- Upstream: https://github.com/lifanchen-simm/TransformerCPI
- License: Apache-2.0
- Reference: Chen, L. et al. "TransformerCPI: improving compound-protein
  interaction prediction by sequence-based deep learning with self-attention
  mechanism and label reversal experiments." *Bioinformatics* 36(16),
  4406-4414 (2020). https://doi.org/10.1093/bioinformatics/btaa524
- Changes made for AdaptiveHIT: added `GPCR/main_integ_trans.py`,
  `GPCR/predict_trans.py`/`predict_trans_em.py`, fixed a metric-computation
  bug in `GPCR/model.py` (precision_recall_curve output was mislabeled as
  tpr/fpr) and modernized `GPCR/Radam.py`'s deprecated PyTorch API calls.

## DeepConv-DTI (`base_model/DeepConv-DTI`)

- Upstream: https://github.com/GIST-CSBL/DeepConv-DTI
- **License: GPL-3.0** -- kept as the pristine, unmodified upstream
  submodule (AdaptiveHIT's own additions exist only as a local patch,
  `patches/DeepConv-DTI.patch`, never committed into a maintained fork) and
  invoked as an independent process (never statically linked into
  AdaptiveHIT's own code), so AdaptiveHIT's own MIT-licensed code is
  unaffected; the GPL-3.0 terms apply to `base_model/DeepConv-DTI` itself.
- Changes made for AdaptiveHIT: added `main_integ_DeepConvDTI.py` (adds an
  HDF5-based streaming data pipeline for datasets too large to fit in
  memory), `predict_deepconv.py`/`predict_deepconv_em.py`.

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
