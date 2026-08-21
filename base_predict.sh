#!/bin/bash
set -e

# ============================================================
# Positional arguments:
#   $1 - data_dir:      Path to the dataset directory (required)
#   $2 - mission:       Name of the prediction run (used for file naming, required)
# ============================================================

# Auto-detect the root directory of AdaptiveHIT
export ADAP_MODEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data_dir="${1:?Error: data_dir (arg1) is required}"
mission="${2:?Error: mission (arg2) is required}"

# Resolve to an absolute path: every step below cd's into another directory,
# so a relative data_dir (e.g. the README's ./dataset/toy_dataset) would
# otherwise be resolved against the wrong working directory.
data_dir="$(cd "$data_dir" && pwd)"

# Pretrained base model checkpoints (shipped via Git LFS)
model_dir_ConPLex="$ADAP_MODEL_ROOT/pretained_models/base_models/conplex.model"
DEVICES_ConPLex="${CUDA_VISIBLE_DEVICES:-0}"
model_dir_DeepConv="$ADAP_MODEL_ROOT/pretained_models/base_models/deep.model"
DEVICES_DeepConv="${CUDA_VISIBLE_DEVICES:-0}"
model_dir_DrugBAN="$ADAP_MODEL_ROOT/pretained_models/base_models/drugban.model"
DEVICES_DrugBAN="${CUDA_VISIBLE_DEVICES:-0}"
model_dir_TransformerCPI="$ADAP_MODEL_ROOT/pretained_models/base_models/trans.model"
DEVICES_TransformerCPI="${CUDA_VISIBLE_DEVICES:-0}"

# Activate Conda and the main environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate adaptivehit

# Use GPU 0 for data preprocessing (can be changed)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ADAP_MODEL_ROOT/scripts/data_processed"

# Step 1: Split and prepare data for base models
echo "Data processing for base models ..."
python process_data_base_model.py "$data_dir" 50000 predict

# Step 2: Generate ID mappings for molecules and proteins
echo "Generating ID mappings for embeddings ..."
python data_process_id.py "$data_dir" "$mission"

# Step 3: Run predictions for all four base models (in parallel)
echo "Starting base model predictions ..."
cd "$ADAP_MODEL_ROOT/bashes/predict"

# DrugBAN
echo "DrugBAN prediction ..."
conda activate drugban
bash integ_screen_drugban_predict.sh \
        "$model_dir_DrugBAN" \
        "$DEVICES_DrugBAN" \
        "$data_dir/data/DrugBAN" \
        "$ADAP_MODEL_ROOT/base_model/DrugBAN" \
        > "$data_dir/data/DrugBAN/DrugBAN.file" 2>&1 &
wait

# DeepConv-DTI
echo "DeepConv-DTI prediction ..."
conda activate DeepConv-DTI
bash integ_screen_deep_predict.sh \
        "$model_dir_DeepConv" \
        "$DEVICES_DeepConv" \
        "$data_dir/data/DeepConv-DTI" \
        "$ADAP_MODEL_ROOT/base_model/DeepConv-DTI" \
        > "$data_dir/data/DeepConv-DTI/DeepConv-DTI.file" 2>&1 &
wait

# ConPLex
echo "ConPLex prediction ..."
conda activate conplex
bash integ_screen_conplex_predict.sh \
        "$model_dir_ConPLex" \
        "$DEVICES_ConPLex" \
        "$data_dir/data/ConPLex" \
        "$ADAP_MODEL_ROOT/base_model/ConPLex_dev" \
        > "$data_dir/data/ConPLex/ConPLex.file" 2>&1 &
wait

# TransformerCPI
echo "TransformerCPI prediction ..."
conda activate transformerCPI
bash integ_screen_trans_predict.sh \
        "$model_dir_TransformerCPI" \
        "$DEVICES_TransformerCPI" \
        "$data_dir/data/TransformerCPI" \
        "$ADAP_MODEL_ROOT/base_model/TransformerCPI" \
        > "$data_dir/data/TransformerCPI/TransformerCPI.file" 2>&1 &

# Wait for all background jobs to finish
wait

# Step 4: Merge and evaluate predictions
cd "$ADAP_MODEL_ROOT/scripts/data_processed"
echo "Merging and evaluating predictions ..."
python result_process_data_integ_and_evalu.py "$data_dir/data" predict
wait