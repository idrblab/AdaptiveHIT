# AdaptiveHIT: Pair-Aware Ensemble Framework for Novel Hit Identification

**AdaptiveHIT** is a pair-aware ensemble framework that adaptively integrates predictions from diverse base models by leveraging their complementary strengths while mitigating individual biases in novel hit identification. Benchmark evaluations demonstrated that this framework outperforms all base models across most scenarios, particularly in identifying bioactive molecules with novel scaffolds. Interpretability analyses further revealed that AdaptiveHIT achieves substantially much higher precision in recognizing ligand binding sites and hot-spot residues.

---

## Installation

```bash
git clone <this-repo-url>
cd AdaptiveHIT

# Set up conda environments for base models and AdaptiveHIT
bash run.sh

# Download external pretrained assets (X-Mol, ESM-2, ProtBert, etc.)
bash pull_external_assets.sh

# Activate the main environment
conda activate adaptivehit
```

- `run.sh` creates 6 conda environments (`adaptivehit`, `conplex`, `transformerCPI`, `DeepConv-DTI`, `drugban`, `xmol`) from each component's own documented dependencies. `xmol` holds the PaddlePaddle 1.8.5 stack the X-Mol molecule featuriser needs; `adaptivehit_train.sh`/`adaptivehit_predict.sh` switch into it automatically for the embedding step. The four base models under `base_model/` are vendored as regular tracked files (no git submodules), so a plain `git clone` is enough. See `NOTICE.md` for each model's upstream commit and the changes applied on top.
- `run.sh` is idempotent: it **skips any conda env that already exists**. If you previously created an env and want to pick up a changed pin (e.g. the CUDA bumps described below), remove it first — `conda env remove -n drugban` — and re-run.
- **GPU note:** the base models pin CUDA builds from their original 2019-2023 releases. `run.sh` pins `cudatoolkit=11.0` / `+cu113` builds, which cover Ampere (A100, sm_80). Newer cards (Ada/Blackwell, e.g. RTX 40/50 series) are **not** covered by these old builds and will fail with "no kernel image is available for execution on the device"; on such hardware run with `CUDA_VISIBLE_DEVICES=-1` (CPU) or rebuild the envs against a newer torch.
- **X-Mol runs on CPU.** PaddlePaddle 1.8.x — the last series with the `paddle.fluid` API X-Mol is written against — only ships CUDA 9.0/10.0 GPU builds, so no GPU build of it supports an A100. Molecule featurisation is a one-off step, so `xmol` installs the CPU build. If you have an older card and install a matching `paddlepaddle-gpu` yourself, set `XMOL_USE_CUDA=true`.
- The `adaptivehit` env installs a **CPU-only** torch (`environment.yml` pins `pytorch::cpuonly` for portability). Swap it for a CUDA build to train or predict on GPU.

Base-model and adaptivehit checkpoints (`pretained_models/`) are bundled via Git LFS and downloaded automatically by the `git clone` above (needs `git-lfs` installed). 

- `pull_external_assets.sh` downloads all five large assets below (X‑Mol weights, ESM‑2, ProtBert, the DUDE decoy set, and the CPI datasets) and extracts each into its target directory. It skips anything already present, so it is safe to re-run after an interrupted download. If it fails (e.g. the host is unreachable), download the files manually and place them in the target directories below.

### Manual Resource Download (optional)

| File Name | Description | File Size | Target Directory | Download Link |
| --- | --- | --- | --- | --- |
| `step_400000_20200326221400.tar` | Required model files for X-MOL | 993.5 MB | `./_ForFeatures/xmol/FT_to_embedding/data/model/step_400000/` | http://47.88.56.212/iTarget/step_400000_20200326221400.tar |
| `esm2_t36_3B_UR50D.pt` | Pretrained ESM-2 weights | 5.28 GB | `./_ForFeatures/esm2/pretrained_esm2_models/` | http://47.88.56.212/iTarget/esm2_t36_3B_UR50D.pt |
| `huggingface.tar` | Pretrained Rostlabprot_bert weights | 1.56 GB | `./base_model/ConPLex_dev/models/` | http://47.88.56.212/iTarget/huggingface.tar |
| `full.tsv` | DUDE full dataset for conplex | 580 MB | `./base_model/ConPLex_dev/dataset/DUDe/` | http://47.88.56.212/iTarget/full.tsv |
| `dataset.tar` | CPI datasets | 798 MB | `./` | http://47.88.56.212/iTarget/dataset.tar |

After downloading the compressed files, extract them into the specified directories.

---

## Quick Start: Prediction with Pretrained Weights (Toy Dataset)

Use the provided toy dataset to run inference with the pretrained models:

```bash
# Predict using the four base models
bash base_predict.sh ./dataset/toy_dataset toy_dataset

# Predict using the AdaptiveHIT (ensemble)
bash adaptivehit_predict.sh ./dataset/toy_dataset toy_dataset
```

- Both scripts assume the pretrained checkpoints are already in place (shipped via Git LFS).
- `base_predict.sh` saves results under `<data_dir>/data/`; `adaptivehit_predict.sh` saves results under `<data_dir>/results_adaptivehit/`.

---

## Retraining on Your Own Data (Toy Dataset)

### 1. Retrain Base Models from Scratch

```bash
bash base_train.sh <data_dir> <mission> <learning_rate> <batch_size> <epochs> <dropout>
```

**Example:**
```bash
bash base_train.sh ./dataset/toy_dataset toy_dataset 0.001 32 50 0.1
```

This will train all four base models on your data (split must be prepared as described in the data preparation guide). The trained models are saved under `<data_dir>/data/*/models/` (where `*` is each base model name).

### 2. Retrain the AdaptiveHIT

```bash
bash adaptivehit_train.sh <data_dir> <mission> <learning_rate> <batch_size> <epochs> <dropout>
```

**Example:**
```bash
bash adaptivehit_train.sh ./dataset/toy_dataset toy_dataset 0.001 32 50 0.1
```

The AdaptiveHIT uses the base models’ predictions and protein/drug representations to build the final model. Results are saved under `<data_dir>/models_adaptivehit/`.

---

## Add or Replace CPI Models for AdaptiveHIT

You can easily extend AdaptiveHIT with your own base models.

**Before running the commands**, you must prepare prediction results from your own model(s) for the training, validation, and test splits of your dataset. These results should be stored in a dedicated directory following the structure:

```
./dataset/<mission>/data/other_model/<model_name>/
```

Under that directory, provide three CSV files:

- `train.csv`
- `val.csv`
- `test.csv`

Each CSV must contain the following columns:

- `Compound_ID`
- `Protein_ID`
- `Predicted_scores_<model_name>` (where `<model_name>` matches the argument passed to `adaptivehit_train_add_predictor.sh`)
- `label_predict_<model_name>`

Ensure that the `Compound_ID` and `Protein_ID` match the identifiers used in your dataset. The order of rows does not need to match the original data, as merging is based on these keys.

After preparing the files, run:

```bash
bash base_predict.sh <data_dir> <mission>   # first, generate predictions for the base models
bash adaptivehit_train_add_predictor.sh <data_dir> <mission> <lr> <batch_size> <epochs> <dropout> <ModelX,ModelY>
```

**Example:**
```bash
bash base_predict.sh ./dataset/Chembl Chembl
bash adaptivehit_train_add_predictor.sh ./dataset/Chembl Chembl 0.001 32 50 0.1 ModelX,ModelY
```

This retrains the AdaptiveHIT using the new set of base models. The comma‑separated list of model names must match the naming convention used in your pipeline.

---

## Amino Acid Mutation Analysis (Interpretability)

To analyze binding site and hot‑spot residues via systematic mutation, you must first prepare the necessary input files.

**Input file preparation:**

For each target protein–ligand pair (e.g., PDB entry `4zts` with ligand `4RK`), create a dedicated directory (e.g., `./dataset/toy_mutate/`) and place the following files:

1. **`test_pdb.csv`** – contains the original protein sequence, SMILES, and binding‑site labels.  
   Example content:
   ```
   PDB_ID,Ligand_Name,Protein,SMILES,label,siteslabel
   4zts,4RK,GAMES...ESASKQS,CCc1ccc(cc1)...[nH]nn1,0,00100...00100
   ```
   The `siteslabel` is a binary string of length equal to the protein sequence, where `1` indicates a binding‑site residue (based on experimental structure or a 5‑Å cutoff).

2. **`<PDB>_<Ligand>_region.csv`** – contains detailed residue‑level information.  
   For example, `4zts_4RK_region.csv`:
   ```
   mutated_position,seq,seq-order,hot_spot_residues,ori_label_5A,Region
   1,G,,0,0,d
   2,A,,0,0,d
   ...
   9,Q,127,0,0,d
   10,W,128,0,0,d
   ...
   ```
   - `mutated_position`: position in the protein sequence (1‑based).
   - `seq`: original amino acid at that position.
   - `seq-order`: optional numbering (e.g., from structure); can be left empty if not used.
   - `hot_spot_residues`: indicator (0/1) for known hot‑spot residues.
   - `ori_label_5A`: original label (0/1) within 5Å of the ligand.
   - `Region`: region classification (e.g., `d` for domain, can be customized).

After preparing these files, run the analysis pipeline:

1. **Generate mutated sequences** for your target proteins:
   ```bash
   python ./scripts/mutate/mutate_generatecsv.py
   ```
   (This script will read the prepared CSV files and produce mutated sequence CSV files.)

2. **Run predictions** on the mutated dataset:
   ```bash
   bash base_predict.sh ./dataset/toy_mutate/mutated_data mutated_data
   bash adaptivehit_predict.sh ./dataset/toy_mutate/mutated_data mutated_data
   ```

3. **Analyze the results** to identify critical residues:
   ```bash
   python ./scripts/mutate/analysis_allresult_mutate.py \
       ./dataset/toy_mutate/mutated_data/results_adaptivehit/test_4zts_4RK.csv_results_4_all.csv \
       ./dataset/toy_mutate/mutated_data/results_adaptivehit \
       ./dataset/toy_mutate \
       4zts_4RK
   ```
   In this example, `4zts_4RK` is the identifier used for that specific PDB‑ligand pair. Replace it with your own target identifier accordingly.

---

## Detailed Parameter Reference

### `base_predict.sh`
| Argument | Description |
|----------|-------------|
| 1        | `data_dir` – path to the dataset directory (must contain `data/` subfolder) |
| 2        | `mission` – a name for this run (used for dataset file naming) |

### `adaptivehit_predict.sh`
| Argument | Description |
|----------|-------------|
| 1        | `data_dir` – same as above |
| 2        | `mission` – same as above |

### `base_train.sh`
| Argument | Description |
|----------|-------------|
| 1        | `data_dir` |
| 2        | `mission` |
| 3        | `learning_rate` – e.g., 0.001 |
| 4        | `batch_size` – e.g., 32 |
| 5        | `epochs` – number of training epochs |
| 6        | `dropout` – dropout rate (e.g., 0.1) |

### `adaptivehit_train.sh`
| Argument | Description |
|----------|-------------|
| 1        | `data_dir` |
| 2        | `mission` – also used as AdaptiveHIT's `--data_subdir` |
| 3–6     | same as `base_train.sh` (`learning_rate`, `batch_size`, `epochs`, `dropout`) |

---

## Tips for Customizing Your Runs

### GPU Device Selection
- All scripts use `export CUDA_VISIBLE_DEVICES=0` by default, which restricts execution to GPU 0.
- To use a different GPU, modify this line in the script before running, or set the environment variable before calling the script:
  ```bash
  export CUDA_VISIBLE_DEVICES=1   # use GPU 1
  bash base_predict.sh ...
  ```
- For multi‑GPU usage, you may need to adjust the `--gpus` arguments inside the individual base‑model scripts (not directly exposed in these wrappers). Refer to each base model’s documentation for distributed training support.

### Pretrained Model Paths
- The pretrained base‑model checkpoints are stored in `./pretained_models/base_models/` and are automatically detected.
- If you have custom checkpoints, you can replace the files in that directory, or modify the `model_dir_*` variables at the top of each script (e.g., `base_predict.sh`) to point to your own paths.

### Hyperparameters for Retraining
- `base_train.sh` and `adaptivehit_train.sh` accept learning rate, batch size, epochs, and dropout as positional arguments.
- For `adaptivehit_train.sh`, AdaptiveHIT uses fixed internal parameters (`batch_size=32`, `lr=0.001`) in the final call to `adaptivehit_run_training.py`. To change these, edit the command directly in the script.
- The `--strategy` and `--fusion_method` options in `adaptivehit_run_training.py` can be changed to experiment with different ensemble approaches (e.g., `probability_only`, `concat`, `gate`). See the script’s help or the AdaptiveHITConfig class for details.

### Where to Find Logs and Outputs
- Each base model writes its stdout/stderr to `$data_dir/data/<model>/<model>.file`.
- AdaptiveHIT training logs are stored in `$data_dir/log_adaptivehit/`.
- Final predictions and evaluation results are placed in `$data_dir/results_adaptivehit/` and `$data_dir/data/end_merged/`, respectively.
- All directories are automatically created if they do not exist.

### Using the Manual Pipeline Step by Step
If you prefer to run each component separately (e.g., for debugging), you can execute the Python commands shown above directly, after ensuring that all prerequisite files (embeddings, merged CSV files) exist. The scripts `result_process_data_integ_and_evalu.py`, `label_ori_merge.py`, and `adaptivehit_run_training.py` each accept command‑line arguments that are self‑explanatory; run them with `--help` for details.

### Note
When retraining ConPLex, its default configuration uses contrastive learning which requires the external DUDE decoy dataset (`base_model/ConPLex_dev/dataset/DUDe/full.tsv`). If not available, set `contrastive: False` in its config file.

---

## License & Attribution

AdaptiveHIT's own code (`scripts/`) is MIT-licensed (`LICENSE`). The four base models under `base_model/` each retain their own upstream license -- DeepConv-DTI is GPL-3.0 and is kept as an independent submodule/process so it never contaminates AdaptiveHIT's own MIT-licensed code; the other three are MIT/Apache-2.0. Full per-submodule provenance and the exact patches applied against each upstream repo are listed in `NOTICE.md`.

**Data provenance:** The `dataset/toy_dataset/` splits are shipped as a ready-to-run worked example. All other datasets, including the `dataset/multisimi/`，`dataset/ChEMBL/`，`dataset/BindingDB/`，`dataset/HUMAN/`，`dataset/HUMAN_cold_pair/` dataset used for the manuscript's robustness experiments, are now fully available for download. You can obtain them by running `pull_external_assets.sh` or by downloading the corresponding archives manually from the links provided in the **Manual Resource Download** section above, and place all extracted files under the `dataset/` directory.

## Citation

A manuscript describing AdaptiveHIT is currently in preparation. Citation details will be added here once published.