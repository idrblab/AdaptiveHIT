#!/bin/bash
set -e

# ============================================================
# Positional arguments:
#   $1 - data_dir:       Path to the dataset directory (required)
#   $2 - mission:        Name of the training run (used for file naming, required)
#   $3 - learning_rate:  Learning rate (default: 0.001)
#   $4 - batch_size:     Batch size (default: 8)
#   $5 - epoch:          Number of training epochs (default: 2)
#   $6 - dropout:        Dropout rate (default: 0.1)
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

# Activate Conda and main environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate adaptivehit

# Data preprocessing (splitting into train/val/test)
cd "$ADAP_MODEL_ROOT/scripts/data_processed"
python process_data_base_model.py "$data_dir" 90000000 random
processed_data_dir="$data_dir/data"

# ----------------------------------------------------------------------
# TransformerCPI retraining
echo "TransformerCPI retraining ..."
conda activate transformerCPI
cd "$processed_data_dir/TransformerCPI"
mkdir -p models results myouts
cd "$ADAP_MODEL_ROOT/base_model/TransformerCPI/GPCR"

# Featurization (generates .npy files for training)
python mol_featurizer_integ_retrain.py \
        "$processed_data_dir" \
        train \
        > "$processed_data_dir/TransformerCPI/mol_featurizer.file" 2>&1 &
wait

# Training (uses GPU 0 by default; change CUDA_VISIBLE_DEVICES if needed)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup python main_integ_trans.py \
        "$processed_data_dir/TransformerCPI/data/train/" \
        "$processed_data_dir/TransformerCPI/data/val/" \
        -r "$learning_rate" \
        -e "$epoch" \
        -D "$dropout" \
        -b "$batch_size" \
        -m "$processed_data_dir/TransformerCPI/models/model-$learning_rate-$batch_size-$dropout-trans.model" \
        -o "$processed_data_dir/TransformerCPI/results/validation_output-$mission-trans.csv" \
        > "$processed_data_dir/TransformerCPI/myouts/myout_$learning_rate-$batch_size-$dropout-trans.file" 2>&1 &
wait

# ----------------------------------------------------------------------
# ConPLex retraining
echo "ConPLex retraining ..."
conda activate conplex
cd "$processed_data_dir/ConPLex"
mkdir -p models results myouts
cd "$ADAP_MODEL_ROOT/base_model/ConPLex_dev"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup python main_integ_conplex.py \
        --exp-id Conplex \
        "$processed_data_dir/ConPLex/data" \
        -r "$learning_rate" \
        -e "$epoch" \
        -D "$dropout" \
        -b "$batch_size" \
        -m "$processed_data_dir/ConPLex/models/model-$learning_rate-$batch_size-$dropout-conplex.model" \
        -o "$processed_data_dir/ConPLex/results/validation_output-$mission-conplex.csv" \
        -c configs/default_config.yaml \
        > "$processed_data_dir/ConPLex/myouts/myout-$learning_rate-$batch_size-$dropout-conplex.file" 2>&1 &
wait

# ----------------------------------------------------------------------
# DeepConv-DTI retraining
echo "DeepConv-DTI retraining ..."
conda activate DeepConv-DTI
cd "$processed_data_dir/DeepConv-DTI"
mkdir -p models results myouts
cd "$ADAP_MODEL_ROOT/base_model/DeepConv-DTI"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup python main_integ_DeepConvDTI.py \
        "$processed_data_dir/DeepConv-DTI/data/train/dti.csv" \
        "$processed_data_dir/DeepConv-DTI/data/train/compound.csv" \
        "$processed_data_dir/DeepConv-DTI/data/train/protein.csv" \
        --validation \
        -n "$processed_data_dir" \
        -i "$processed_data_dir/DeepConv-DTI/data/val/dti.csv" \
        -d "$processed_data_dir/DeepConv-DTI/data/val/compound.csv" \
        -t "$processed_data_dir/DeepConv-DTI/data/val/protein.csv" \
        -W \
        -c 512 128 \
        -w 10 15 20 25 30 \
        -p 128 \
        -f 128 \
        -r "$learning_rate" \
        -n "$processed_data_dir" \
        -v Convolution \
        -l 2500 \
        -V morgan_fp_r2 \
        -L 2048 \
        -D "$dropout" \
        -a elu \
        -F 128 \
        -b "$batch_size" \
        -y 0.0001 \
        -o "$processed_data_dir/DeepConv-DTI/results/validation_output-$mission-deepconv.csv" \
        -m "$processed_data_dir/DeepConv-DTI/models/model-$learning_rate-$batch_size-$dropout-deepconv" \
        -e "$epoch"  \
        > "$processed_data_dir/DeepConv-DTI/myouts/myout_$learning_rate-$batch_size-$dropout-deepconv.file" 2>&1 &
wait

# ----------------------------------------------------------------------
# DrugBAN retraining
echo "DrugBAN retraining ..."
conda activate drugban
cd "$processed_data_dir/DrugBAN"
mkdir -p models results myouts
cd "$ADAP_MODEL_ROOT/base_model/DrugBAN"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup python main_integ_drugban.py "$processed_data_dir/DrugBAN/data" \
        -r "$learning_rate" \
        -e "$epoch" \
        -b "$batch_size" \
        -m "$processed_data_dir/DrugBAN/models/model-$learning_rate-$batch_size-$dropout-drugban.model" \
        -o "$processed_data_dir/DrugBAN/results/validation_output-$mission-drugban.csv" \
        --data Drugban \
        > "$processed_data_dir/DrugBAN/myouts/myout-$learning_rate-$batch_size-$dropout-drugban.file" 2>&1 &
wait

echo "All base models trained successfully."