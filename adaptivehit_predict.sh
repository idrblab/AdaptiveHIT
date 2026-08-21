#!/bin/bash
set -e

# ============================================================
# Positional arguments:
#   $1 - data_dir:      Path to the dataset directory (required)
#   $2 - mission:       Name of the prediction run (required)
# ============================================================

export ADAP_MODEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${1:?Error: data_dir is required}"
mission="${2:?Error: mission is required}"

# Resolve to an absolute path: every step below cd's into another directory,
# so a relative data_dir (e.g. the README's ./dataset/toy_dataset) would
# otherwise be resolved against the wrong working directory.
data_dir="$(cd "$data_dir" && pwd)"

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate adaptivehit

# Use GPU 0 for all steps
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ADAP_MODEL_ROOT/scripts/data_processed"

# Step 1: Generate molecule embeddings (X-Mol)
echo "Generating molecule embeddings with X-Mol ..."
bash "$ADAP_MODEL_ROOT/_ForFeatures/xmol/bashes/run_emb_all_new.sh" \
        "$mission" \
        "$data_dir"
python prebuild_xmol_cache.py \
        "$data_dir" \
        "$mission" \
        "$ADAP_MODEL_ROOT/_ForFeatures/xmol/FT_to_embedding/data/for_output"

# Step 2: Generate protein embeddings (ESM-2)
echo "Generating protein embeddings with ESM-2 ..."
bash "$ADAP_MODEL_ROOT/_ForFeatures/esm2/bashes/template_esm2_t36_3B_UR50D.sh" \
        "$data_dir" \
        "$mission"

# Step 3: Merge ground-truth labels with base model predictions
echo "Merging original labels ..."
python label_ori_merge.py "$data_dir" predict "$mission"
wait

# Step 4: Run the AdaptiveHIT predictor
echo "Running AdaptiveHIT prediction ..."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ADAP_MODEL_ROOT/scripts"
python predict.py \
        --input_dir "$data_dir/data/end_merged" \
        --model_dir "$ADAP_MODEL_ROOT/pretained_models" \
        --weights_dir "$data_dir/weights"  \
        --output_dir "$data_dir/results_adaptivehit" \
        --dataset_name "$mission" \
        --mode 4 \
        --models_mode all \
        --base_models TransformerCPI DeepConv-DTI ConPLex DrugBAN \
        --strategies adaptivehit_full_esm2_xmol_prob_attention \
        --data_subdir "$mission"

echo "AdaptiveHIT prediction completed. Results saved in $data_dir/results_adaptivehit/"