#!/bin/bash
set -e

# ============================================================
# Positional arguments:
#   $1 - data_dir:       Path to the dataset directory (required)
#   $2 - mission:        Name of the training run (required)
#   $3 - learning_rate:  Learning rate for base model retraining (default: 0.001)
#   $4 - batch_size:     Batch size for base model retraining (default: 8)
#   $5 - epoch:          Epochs for base model retraining (default: 2)
#   $6 - dropout:        Dropout for base model retraining (default: 0.1)
# ============================================================

export ADAP_MODEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data_dir="${1:?Error: data_dir is required}"
mission="${2:?Error: mission is required}"
learning_rate="${3:-0.001}"
batch_size="${4:-8}"
epoch="${5:-2}"
dropout="${6:-0.1}"

# Resolve to an absolute path: every step below cd's into another directory,
# so a relative data_dir (e.g. the README's ./dataset/toy_dataset) would
# otherwise be resolved against the wrong working directory.
data_dir="$(cd "$data_dir" && pwd)"

# Paths to the best models obtained from base_train.sh
model_dir_ConPLex="$data_dir/data/ConPLex/models/model-$learning_rate-$batch_size-$dropout-conplex_best.model"
model_dir_DeepConv="$data_dir/data/DeepConv-DTI/models/model-$learning_rate-$batch_size-$dropout-deepconv_best.model"
model_dir_DrugBAN="$data_dir/data/DrugBAN/models/model-$learning_rate-$batch_size-$dropout-drugban_best.model"
model_dir_TransformerCPI="$data_dir/data/TransformerCPI/models/model-$learning_rate-$batch_size-$dropout-trans_best.model"

# All base models will run on the same GPU (set to 0; modify if needed)
DEVICES_ConPLex="${CUDA_VISIBLE_DEVICES:-0}"
DEVICES_DeepConv="${CUDA_VISIBLE_DEVICES:-0}"
DEVICES_DrugBAN="${CUDA_VISIBLE_DEVICES:-0}"
DEVICES_TransformerCPI="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate adaptivehit

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ADAP_MODEL_ROOT/scripts/data_processed"

# 1. Generate ID mappings for embeddings
echo "Generating ID mappings ..."
python data_process_id.py "$data_dir" "$mission"

# 2. Generate molecule embeddings (X-Mol)
echo "Generating molecule embeddings with X-Mol ..."
bash "$ADAP_MODEL_ROOT/_ForFeatures/xmol/bashes/run_emb_all_new.sh" "$mission" "$data_dir"
python prebuild_xmol_cache.py \
        "$data_dir" \
        "$mission" \
        "$ADAP_MODEL_ROOT/_ForFeatures/xmol/FT_to_embedding/data/for_output"

# 3. Generate protein embeddings (ESM-2)
echo "Generating protein embeddings with ESM-2 ..."
bash "$ADAP_MODEL_ROOT/_ForFeatures/esm2/bashes/template_esm2_t36_3B_UR50D.sh" "$data_dir" "$mission"

# 4. Run base model predictions (using the trained models)
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
wait

# 5. Merge predictions and labels
cd "$ADAP_MODEL_ROOT/scripts/data_processed"
echo "Post-processing predictions ..."
python result_process_data_integ_and_evalu.py "$data_dir/data" predict
wait

echo "Merging original labels ..."
python label_ori_merge.py "$data_dir" predict "$mission"
wait

# 6. Train AdaptiveHIT
echo "Training AdaptiveHIT ..."
cd "$data_dir"
mkdir -p models_adaptivehit results_adaptivehit log_adaptivehit weights
cd "$ADAP_MODEL_ROOT/scripts"

# Note: AdaptiveHIT training uses its own fixed parameters (batch_size=32, lr=0.001, etc.)
# To change them, modify the command below.
conda activate adaptivehit
python adaptivehit_run_training.py \
--dataset "$mission" \
--data_subdir "$mission" \
--input_dir "$data_dir/data/end_merged" \
--output_dir "$data_dir/models_adaptivehit/adaptivehit_full_esm2_xmol_prob_attention-$batch_size-$learning_rate" \
--strategy full_representations \
--fusion_method prob_attention \
--epochs "$epoch" \
--batch_size 32 \
--lr 0.001 \
--use_weight_balance \
--model_names "TransformerCPI,ConPLex,DeepConv-DTI,DrugBAN" \
> "$data_dir/log_adaptivehit/adaptivehit_full_esm2_xmol_prob_attention-$batch_size-$learning_rate-$mission.output" 2>&1 &
wait

echo "Meta-training completed."