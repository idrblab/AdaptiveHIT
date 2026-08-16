# AdaptiveHIT: Pair-Aware Ensemble Framework for Novel Hit Identification

**AdaptiveHIT** is a pair-aware ensemble framework that adaptively integrates predictions from diverse base models by leveraging their complementary strengths while mitigating individual biases in novel hit identification. Benchmark evaluations demonstrated that this framework outperforms all base models across most scenarios, particularly in identifying bioactive molecules with novel scaffolds. Interpretability analyses further revealed that AdaptiveHIT achieves substantially much higher precision in recognizing ligand binding sites and hot-spot residues.

---

## Installation

```bash
git clone <this-repo-url>
cd AdaptiveHIT
bash run.sh
conda activate adaptivehit
```

A plain `git clone` brings in everything -- the 4 base models under `base_model/` are vendored directly into this repository (not git submodules), so there's no separate init step. `run.sh` creates 5 conda envs (`adaptivehit`, `conplex`, `transformerCPI`, `DeepConv-DTI`, `drugban`) from each base model's own documented dependencies. It's idempotent -- safe to re-run. If a base model's env fails to solve (a few pin old CUDA/package versions from their original 2019-2023 releases), see the comments in `run.sh` or that model's own `README.md`.

Base-model and MetaLearner checkpoints (`checkpoints/`) and the X-Mol pretrained weights (`data_adapter/xmol_weights/`) are bundled via Git LFS and downloaded automatically by the `git clone` above (needs `git-lfs` installed). **ESM-2 is not bundled** -- it's fetched on first use, see Quickstart below.

---

## Quickstart (toy dataset)

Generate ESM-2/X-Mol embeddings for the shipped `data/toy_dataset/`, then run the pretrained MetaLearner:

```bash
python data_adapter/generate_esm2_embeddings.py \
    --prots_csv data/toy_dataset/id/toy_dataset_prots.csv \
    --datatype toy_dataset --output_dir data/embeddings/esm2_t36_3B_UR50D

python data_adapter/xmol_embed.py \
    --drugs_csv data/toy_dataset/id/toy_dataset_drugs.csv \
    --datatype toy_dataset --output_dir data/embeddings/xmol
python data_adapter/prebuild_xmol_cache.py data/toy_dataset toy_dataset data/embeddings/xmol

python meta_learner/predict.py \
    --input_dir data/toy_dataset/end_merged \
    --output_dir out/ \
    --data_subdir toy_dataset \
    --protein_emb_dir data/embeddings/esm2_t36_3B_UR50D \
    --drug_emb_dir data/embeddings/xmol \
    --eval
```

| Argument | Description |
|---|---|
| `--prots_csv` / `--drugs_csv` | protein CSV (`protid` + `sequence` columns) / drug CSV (SMILES column) |
| `--datatype` | subdirectory name under `--output_dir` |
| `--input_dir` | merged prediction+label CSVs -- see [Running on Your Own Data](#running-on-your-own-data) for how to produce these |
| `--output_dir` | where predictions/metrics are written |
| `--model_dir` | defaults to the shipped checkpoint `checkpoints/meta/meta_full_esm2_xmol_prob_attention/` |
| `--protein_emb_dir` / `--drug_emb_dir` | **must be passed explicitly**, see note below |
| `--eval` | also compute AUC/F1/MCC against the label column |

> `--protein_emb_dir`/`--drug_emb_dir` look optional in `meta_config.py`'s defaults but aren't: the shipped checkpoint's saved config carries the original training cluster's absolute paths, and `predict.py` only overrides those when the flags are passed explicitly. Omitting them fails with `FileNotFoundError: Global feature library not found: ...`.

`esm2_t36_3B_UR50D` (~5.4GB) downloads automatically on first run via `fair-esm`, cached to `~/.cache/torch/hub/checkpoints/`. On an unreliable connection, fetch it yourself and let `fair-esm` find it already cached:
```bash
curl -fL -C - -o ~/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt \
    https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt
```

---

## Running on Your Own Data

The `pipeline/*.sh` scripts below `cd` into other directories internally, so pass **absolute paths** for your own data/output locations (e.g. `$(pwd)/mydata` from the repo root). The Python scripts above don't have this restriction.

**0. Predict with the shipped base-model checkpoints** on your own test set:
```bash
pipeline/predict/integ_screen_conplex_predict.sh checkpoints/base_models/model-0.0005-64-0.1-conplex-858.model '' <testset_dir> $(pwd)/base_model/ConPLex_dev
pipeline/predict/integ_screen_drugban_predict.sh checkpoints/base_models/model-0.0005-64-0.1-drugban-940.model '' <testset_dir> $(pwd)/base_model/DrugBAN
pipeline/predict/integ_screen_trans_predict.sh   checkpoints/base_models/model-0.0005-128-0.1-trans-946.model '' <testset_dir> $(pwd)/base_model/TransformerCPI
pipeline/predict/integ_screen_deep_predict.sh    checkpoints/base_models/model-0.0005-64-0.1-deep-969.model '' <testset_dir> $(pwd)/base_model/DeepConv-DTI
```

| Argument | Description |
|---|---|
| 1 | base-model checkpoint (shipped under `checkpoints/base_models/`) |
| 2 | CUDA device id, or `''` for CPU |
| 3 | `<testset_dir>`, expected to contain `<testset_dir>/data/<subset>/*.csv` |
| 4 | absolute path to the base model's directory under `base_model/` |

Merge the 4 base models' predictions with your ground-truth labels:
```bash
python data_adapter/result_process_data_integ_and_evalu.py <testset_dir>/data predict
python data_adapter/label_ori_merge.py <testset_dir> predict <datatype>
```

Or run the full orchestration in one call: `pipeline/meta_train.sh $(pwd)` (training-time) / `pipeline/meta_predict.sh $(pwd)` (inference-time) -- both are internal cluster-authored scripts with more hardcoded assumptions than the ones above, so expect to read/adjust them before use.

**1. Retrain the base models** from scratch on your own train/val/test split:
```bash
pipeline/base_retrain.sh <data_dir> $(pwd)
```
ConPLex defaults to `contrastive: True` (`configs/default_config.yaml`), which needs the external DUDE decoy dataset (`base_model/ConPLex_dev/dataset/DUDe/full.tsv`, not shipped). Supply it yourself, or pass a config with `contrastive: False`.

**2. Retrain the MetaLearner** on your own merged Phase 0 output:
```bash
python meta_learner/meta_run_training.py \
    --input_dir <phase0_merged_dir> --output_dir <outputs_dir> \
    --data_subdir <name> --strategy probability_only
```
Then point `predict.py --model_dir` at your own retrained checkpoint instead of the shipped one.

---

## Adding or Replacing a Base Model

The MetaLearner's fusion network and the data-merge scripts size themselves off however many base models you tell them about -- nothing is hardcoded to exactly 4.

**1. Add the model's code** under `base_model/<Name>/`, following the pattern of the existing 4: a `main_integ_<name>.py` training entrypoint and `predict_<name>.py` inference entrypoint (see any existing base model for the argument style, which varies by architecture), plus `pipeline/train/retrain_<name>_integ.sh` / `pipeline/predict/integ_screen_<name>_predict.sh` wrappers.

**2. Match the output contract.** `predict_<name>.py` must write one CSV per dataset to `<testset_dir>/results_meta/<dataset>.csv` with columns `Compound_ID, Protein_ID, Predicted_scores, label_predict` (add `label_original` too when run in train/eval mode) -- this is the only thing `data_adapter/result_process_data_integ_and_evalu.py` and the MetaLearner's data loader (`meta_data_loader.py`) actually require.

**3. Register the model name**, all overridable without code edits:

| Where | How |
|---|---|
| `meta_learner/meta_config.py` `model_names` | pass your own list, or edit the default (`~line 108`) |
| `meta_learner/predict.py` | `--base_models <name1> <name2> ...` |
| `data_adapter/result_process_data_integ_and_evalu.py` | trailing CLI args, anchor model first: `python result_process_data_integ_and_evalu.py <data_dir> <mode> <Model1> <Model2> ...` |
| `data_adapter/label_ori_merge.py` | 4th CLI arg, the model count: `python label_ori_merge.py <data_dir> <mode> <datatype> <N>` |

**4. Retrain the MetaLearner** (`meta_learner/meta_run_training.py`) -- swapping or adding a base model changes the fusion network's input distribution, so the shipped checkpoint no longer applies.

---

## License & Attribution

AdaptiveHIT's own code (`meta_learner/`, `data_adapter/`, `pipeline/`) is MIT-licensed (`LICENSE`). The four base models under `base_model/` are vendored from their original authors' repositories (not forks) and each retains its own upstream license -- DeepConv-DTI is GPL-3.0 and is invoked as an independent process so it never contaminates AdaptiveHIT's own MIT-licensed code; the other three are MIT/Apache-2.0. Full per-model provenance (upstream commit, license, changes made) is listed in `NOTICE.md`.

**Data provenance:** the `data/toy_dataset/` splits are shipped as a ready-to-run worked example; the script recovering their exact derivation from the original raw ChEMBL source table was not available at release time (see `NOTICE.md`). A larger 5-fold cross-validation dataset used for the manuscript's robustness experiments is not included here -- contact the corresponding author for access.

## Citation

A manuscript describing AdaptiveHIT is currently in preparation. Citation details will be added here once published.
