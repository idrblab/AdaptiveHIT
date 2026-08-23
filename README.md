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

`run.sh` creates 6 conda envs (`adaptivehit`, `conplex`, `transformerCPI`,
`DeepConv-DTI`, `drugban`, `xmol`); the pipeline scripts switch between them
automatically. Both scripts are idempotent and skip whatever already exists — to
pick up a changed pin, remove that env first (`conda env remove -n drugban`) and
re-run.

`base_model/` is vendored as regular tracked files, not submodules, so a plain
`git clone` is enough; `pretained_models/` arrives via Git LFS (needs
`git-lfs`). Upstream commits and local changes are recorded in `NOTICE.md`.

**GPU coverage differs per env**, since each pins the CUDA build of its original
2019-2023 release:

| Env | Build | Newest GPU covered |
| --- | --- | --- |
| `conplex`, `transformerCPI` | torch `+cu113` | Ampere (sm_80) |
| `drugban` | torch 1.7.1 + `cudatoolkit=11.0` | Ampere (sm_80) |
| `DeepConv-DTI` | `nvidia-tensorflow` (CUDA 11.8) | Hopper (sm_90); Blackwell works via PTX JIT |
| `xmol` | PaddlePaddle 1.8.5, **CPU only** | — no GPU build of this series reaches Ampere |
| `adaptivehit` | torch + `pytorch-cuda=12.1` | Hopper (sm_90) |

On a newer card than its build covers, a step fails with "no kernel image is
available for execution on the device"; run it with `CUDA_VISIBLE_DEVICES=-1` or
rebuild that env against a newer torch (for Blackwell, a cu128 build).

`xmol` is the one env with no GPU option at all: X-Mol is written against the
`paddle.fluid` API, which Paddle dropped after 1.8.x, and every
`paddlepaddle-gpu` 1.8.x wheel is either `.post97` (CUDA 9.0) or `.post107`
(CUDA 10.0) — neither reaches sm_80. No Paddle release has both the API X-Mol
needs and support for a modern GPU. Molecule featurisation is a one-off step, so
this costs little; set `XMOL_USE_CUDA=true` only if you have a pre-Ampere card
and install a matching `paddlepaddle-gpu` yourself.

### Manual Resource Download (optional)

`pull_external_assets.sh` fetches all five from the mirror. To download by hand,
prefer the upstream where one exists — it is the authoritative copy, and the
mirror is a byte-identical duplicate of it.

| File Name | Description | Size | Target Directory | Upstream | Mirror |
| --- | --- | --- | --- | --- | --- |
| `step_400000_20200326221400.tar` | X-MOL pretrained weights | 993.5 MB | `./_ForFeatures/xmol/FT_to_embedding/data/model/step_400000/` | [bm2-lab/X-MOL](https://github.com/bm2-lab/X-MOL) → "Pre_trained X-MOL" ([OneDrive](https://1drv.ms/u/s!BIa_gVKaCDngi2S994lMsp-Y3TWK?e=l5hbxi)) | [link](http://47.88.56.212/adaptivehit/step_400000_20200326221400.tar) |
| `esm2_t36_3B_UR50D.pt` | Pretrained ESM-2 weights | 5.28 GB | `./_ForFeatures/esm2/pretrained_esm2_models/` | [Meta AI](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt) | [link](http://47.88.56.212/adaptivehit/esm2_t36_3B_UR50D.pt) |
| `huggingface.tar` | ProtBert weights (optional, see below) | 1.56 GB | `./base_model/ConPLex_dev/models/` | [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert) | [link](http://47.88.56.212/adaptivehit/huggingface.tar) |
| `full.tsv` | DUDE decoy set, for retraining ConPLex | 580 MB | `./base_model/ConPLex_dev/dataset/DUDe/` | [ConPLex authors](https://cb.csail.mit.edu/conplex/data/full.tsv) | [link](http://47.88.56.212/adaptivehit/full.tsv) |

Extract each archive into its target directory.

Only the X-MOL weights are strictly required; ESM-2 is needed for protein
embeddings, and `full.tsv` only to retrain ConPLex. `huggingface.tar` is
optional because `ConPLex_dev/src/featurizers/protein.py` falls back to the Hub
when the local copy is absent — download it only for a machine that cannot reach
huggingface.co.

The CPI datasets used for the manuscript's benchmarks (ChEMBL, Multisimi,
BindingDB, HUMAN, HUMAN_cold_pair) are released with this work rather than
mirrored from a third party; `pull_external_assets.sh` fetches them, or take
[dataset.tar](http://47.88.56.212/adaptivehit/dataset.tar) (798 MB) and extract
it at the repository root. Neither Quick Start below needs them —
`dataset/toy_dataset/` ships in this repository.

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