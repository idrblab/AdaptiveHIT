#!/bin/bash
data_dir=$1
DEVICES=$2
learning_rate=$3
batch_size=$4
epoch=$5
mode_train=$6
model_env_dir=$7  # path to the ConPLex_dev submodule (base_model/ConPLex_dev)
dropout=0.1

source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate conplex
conda env list


echo "data_dir: $data_dir/data"
echo "learning-rate: $learning_rate"
echo "batch-size: $batch_size"
echo "dropout: $dropout"
echo "----------------"
export CUDA_VISIBLE_DEVICES=$DEVICES
python main_integ_conplex.py --exp-id Conplex $data_dir/data/$data_sub -r $learning_rate -e $epoch -D $dropout -b $batch_size -m $data_dir/models/$data_sub/model-$learning_rate-$batch_size-$dropout-conplex.model -o $data_dir/results/$data_sub/validation_output.csv -c $model_env_dir/configs/default_config.yaml > $data_dir/myouts/$data_sub/myout-$learning_rate-$batch_size-$dropout.file 2>&1 &
wait

