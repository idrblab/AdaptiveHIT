#!/bin/bash
set -e

# ============================================================
# Positional arguments:
#   $1 - data_dir:           Path to the dataset directory (required)
#   $2 - mission:            Name of the training run (required)
#   $3 - learning_rate:      Learning rate (default: 0.001)
#   $4 - batch_size:         Batch size (default: 8)
#   $5 - epoch:              Epochs (default: 2)
#   $6 - dropout:            Dropout (default: 0.1)
#   $7 - Other_models_list:  Comma-separated list of additional model names,
#                            e.g., "ModelX,ModelY" (required)
# ============================================================

export ADAP_MODEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data_dir="${1:?Error: data_dir is required}"
mission="${2:?Error: mission is required}"
learning_rate="${3:-0.001}"
batch_size="${4:-8}"
epoch="${5:-2}"
dropout="${6:-0.1}"
Other_models_list="${7:?Error: other_models_list is required}"

# Resolve to an absolute path: every step below cd's into another directory,
# so a relative data_dir (e.g. the README's ./dataset/toy_dataset) would
# otherwise be resolved against the wrong working directory.
data_dir="$(cd "$data_dir" && pwd)"

# Use pretrained base models (shipped) for this procedure
model_dir_ConPLex="$ADAP_MODEL_ROOT/pretained_models/base_models/conplex.model"
model_dir_DeepConv="$ADAP_MODEL_ROOT/pretained_models/base_models/deep.model"
model_dir_DrugBAN="$ADAP_MODEL_ROOT/pretained_models/base_models/drugban.model"
model_dir_TransformerCPI="$ADAP_MODEL_ROOT/pretained_models/base_models/trans.model"

DEVICES_ConPLex="${CUDA_VISIBLE_DEVICES:-0}"
DEVICES_DeepConv="${CUDA_VISIBLE_DEVICES:-0}"
DEVICES_DrugBAN="${CUDA_VISIBLE_DEVICES:-0}"
DEVICES_TransformerCPI="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate drugban  # temporarily use drugban env for data processing

# 1. Generate embeddings
echo "Generating molecule embeddings with X-Mol ..."
conda activate xmol
bash "$ADAP_MODEL_ROOT/_ForFeatures/xmol/bashes/run_emb_all_new.sh" "$mission" "$data_dir"
conda activate drugban
python prebuild_xmol_cache.py \
        "$data_dir" \
        "$mission" \
        "$ADAP_MODEL_ROOT/_ForFeatures/xmol/FT_to_embedding/data/for_output"

echo "Generating protein embeddings with ESM-2 ..."
conda activate esm2
bash "$ADAP_MODEL_ROOT/_ForFeatures/esm2/bashes/template_esm2_t36_3B_UR50D.sh" "$data_dir" "$mission"

# 2. Merge original labels
conda activate drugban
echo "Merging original labels ..."
python label_ori_merge.py "$data_dir" predict "$mission"
wait

# 3. Merge predictions from other models (user-provided)
echo "Merging other model predictions ..."
python merge_predictions.py \
    --integrated_dir "$data_dir/data/end_merged" \
    --model_root "$data_dir/data/other_model_results" \
    --model_names "$Other_models_list" \
    --output_dir "$data_dir/data/end_merged_add"

# 4. Train AdaptiveHIT with the extended model list
echo "Training AdaptiveHIT with additional models ..."
cd "$data_dir"
mkdir -p models_adaptivehit results_adaptivehit log_adaptivehit weights
cd "$ADAP_MODEL_ROOT/scripts"

python adaptivehit_run_training.py \
--dataset "$mission" \
--data_subdir "$mission" \
--input_dir "$data_dir/data/end_merged_add" \
--output_dir "$data_dir/models_adaptivehit/adaptivehit_full_esm2_xmol_prob_attention-$batch_size-$learning_rate" \
--strategy full_representations \
--fusion_method prob_attention \
--epochs "$epoch" \
--batch_size 32 \
--lr 0.001 \
--use_weight_balance \
--model_names "TransformerCPI,ConPLex,DeepConv-DTI,DrugBAN,$Other_models_list" \
> "$data_dir/log_adaptivehit/adaptivehit_full_esm2_xmol_prob_attention-$batch_size-$learning_rate-$mission.output" 2>&1 &
wait

echo "Meta-training with new predictors completed."