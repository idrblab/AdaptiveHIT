# AdaptiveHIT

**AdaptiveHIT** is a pair-aware ensemble framework for novel hit
identification, integrating four compound-protein interaction (CPI)
prediction models -- **DeepConv-DTI**, **TransformerCPI**, **DrugBAN**, and
**ConPLex** -- through an adaptive weighting strategy driven by a
probability-guided attention fusion network. For each query, the four base
models independently output an interaction probability, which is combined
with **X-Mol** and **ESM-2** sequence representations of the compound and
protein to adaptively weight each model's contribution and compute the final
ensemble score. AdaptiveHIT outperforms every individual base model at
discovering bioactive molecules with novel structural scaffolds, and
interpretability analysis shows it identifies proteins' ligand binding sites
and hot-spot residues with markedly higher precision than any single model.

A live demo server is available; see the manuscript (in preparation) for
full methodology and benchmark results.

## Pipeline

```
base-model training  ──►  base-model prediction   ──►  merge predictions   ──►  meta-learner  ──►  ensemble
(pipeline/base_retrain.sh)  + embedding extraction      + labels/IDs           training           prediction
                             (pipeline/meta_train.sh,    (data_adapter/         (meta_learner/     (meta_learner/
                              meta_predict.sh)            result_process_*,     meta_run_          predict.py)
                                                           label_ori_merge.py)   training.py)
```

1. **Train the 4 base models** on your own train/val/test split
   (`pipeline/base_retrain.sh`) -- each base model is a separate git
   submodule under `base_model/`, trained via its own
   `main_integ_*.py` entrypoint.
2. **Run each trained base model's predictions** and extract ESM-2/X-Mol
   representations for every compound/protein (`pipeline/meta_train.sh`,
   `pipeline/predict/integ_screen_*_predict.sh`).
3. **Merge** the four base models' predictions with the original labels
   (`data_adapter/result_process_data_integ_and_evalu.py`,
   `data_adapter/label_ori_merge.py`).
4. **Train the MetaLearner** ensemble on top of the merged predictions +
   representations (`meta_learner/meta_run_training.py`).
5. **Run ensemble inference** (`meta_learner/predict.py`) -- defaults to the
   shipped pretrained checkpoint (`checkpoints/meta/`), or point it at your
   own retrained one.

`data/toy_dataset/` is a small, ready-to-run worked example covering steps
3-5 (base-model predictions are already merged; see
`data/toy_dataset/results_meta/` for expected output).

## Repository layout

```
AdaptiveHIT/
├── base_model/            # 4 base CPI models (git submodules, forked + patched)
│   ├── ConPLex_dev/
│   ├── DrugBAN/
│   ├── TransformerCPI/
│   └── DeepConv-DTI/
├── meta_learner/           # MetaLearner ensemble: training + inference
│   └── predict.py          # ensemble inference entrypoint (see Quickstart)
├── data_adapter/            # data-format adapters between the shared dataset
│   │                        # schema and each base model's native input format
│   ├── xmol_weights/        # trimmed pretrained X-Mol weights (Git LFS)
│   └── xmol_embed.py        # pure-PyTorch X-Mol embedder (no PaddlePaddle needed)
├── pipeline/                # end-to-end orchestration shell scripts
├── checkpoints/             # shipped pretrained weights (Git LFS)
│   ├── base_models/         # the 4 trained base-model checkpoints
│   └── meta/                # the trained MetaLearner checkpoint
├── data/toy_dataset/         # small worked example (train/val/test + expected output)
├── NOTICE.md                 # per-submodule license/attribution
└── environment.yml           # conda env for meta_learner/ + data_adapter/
```

Each `base_model/*` submodule keeps its own conda environment (see its own
`environment.yml`/`README.md`) since DeepConv-DTI needs a legacy
TensorFlow-1/Keras stack while the other three are PyTorch-based -- they are
never imported directly into `meta_learner/`, only invoked as separate
processes via `pipeline/*.sh`.

## Quickstart

```bash
git clone --recurse-submodules <this-repo-url>
cd AdaptiveHIT
conda env create -f environment.yml
conda activate adaptivehit

# Ensemble inference with the shipped pretrained checkpoint, against the
# worked example's already-merged data:
cd meta_learner
python predict.py \
    --input_dir ../data/toy_dataset \
    --output_dir /tmp/adaptivehit_out \
    --eval
```

`predict.py --model_dir`/`--strategies` default to the shipped pretrained
checkpoint (`checkpoints/meta/meta_full_esm2_xmol_prob_attention/`); point
them at your own retrained checkpoint to use it instead.

### Retraining on your own data

See `pipeline/base_retrain.sh` (per-base-model training),
`pipeline/meta_train.sh` (full base-predict -> merge -> meta-train
orchestration), and `pipeline/meta_predict.sh` (full pipeline ending in
ensemble prediction) for the exact commands, each parameterized by your own
`data_dir`/`model_env_dir` (this repo's own root).

Embedding *generation* for a brand-new dataset needs:
- **ESM-2** protein embeddings: the standard `fair-esm` package (`pip
  install fair-esm`), no vendoring needed.
- **X-Mol** drug embeddings: `data_adapter/xmol_embed.py` reuses the shipped
  trimmed weights (a pure-PyTorch reimplementation of X-Mol's architecture,
  no PaddlePaddle required) -- run it directly (`python xmol_embed.py
  --drugs_csv ... --datatype ... --output_dir ...`) to produce the
  per-drug `.npy` files `data_adapter/prebuild_xmol_cache.py` expects.

## License

AdaptiveHIT's own code is MIT-licensed (`LICENSE`). The four base models
under `base_model/` each retain their own upstream license -- see
`NOTICE.md`.

## Citation

A manuscript describing AdaptiveHIT is currently in preparation. Citation
details will be added here once published.
