# AdaptiveHIT: Pair-Aware Ensemble Framework for Novel Hit Identification

**AdaptiveHIT** is a pair-aware ensemble framework that adaptively integrates predictions from diverse base models by leveraging their complementary strengths while mitigating individual biases in novel hit identification. Benchmark evaluations demonstrated that this framework outperforms all base models across most scenarios, particularly in identifying bioactive molecules with novel scaffolds. Interpretability analyses further revealed that AdaptiveHIT achieves substantially much higher precision in recognizing ligand binding sites and hot-spot residues.

![Model Architecture](./assets/AdaptiveHIT.png)

---

## Installation

```bash
git clone --recurse-submodules <this-repo-url>
cd AdaptiveHIT
bash run.sh
conda activate adaptivehit
```

`run.sh` applies AdaptiveHIT's patches to each `base_model/*` submodule and creates 5 conda envs (`adaptivehit`, `conplex`, `transformerCPI`, `DeepConv-DTI`, `drugban`) from each submodule's own documented dependencies. It's idempotent -- safe to re-run. If a base model's env fails to solve (a few pin old CUDA/package versions from their original 2019-2023 releases), see the comments in `run.sh` or that submodule's own `README.md`.

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

`esm2_t36_3B_UR50D` (~5.4GB) downloads automatically on first run via `fair-esm`, cached to `~/.cache/torch/hub/checkpoints/`. That download isn't resumable; on an unreliable connection, fetch it yourself and let `fair-esm` find it already cached:
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
| 4 | absolute path to the base model's submodule directory |

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

## License & Attribution

AdaptiveHIT's own code (`meta_learner/`, `data_adapter/`, `pipeline/`) is MIT-licensed (`LICENSE`). The four base models under `base_model/` each retain their own upstream license -- DeepConv-DTI is GPL-3.0 and is kept as an independent submodule/process so it never contaminates AdaptiveHIT's own MIT-licensed code; the other three are MIT/Apache-2.0. Full per-submodule provenance and the exact patches applied against each upstream repo are listed in `NOTICE.md`.

**Data provenance:** the `data/toy_dataset/` splits are shipped as a ready-to-run worked example; the script recovering their exact derivation from the original raw ChEMBL source table was not available at release time (see `NOTICE.md`). A larger 5-fold cross-validation dataset used for the manuscript's robustness experiments is not included here -- contact the corresponding author for access.

## Citation

A manuscript describing AdaptiveHIT is currently in preparation. Citation details will be added here once published.
