#!/bin/bash
set -e

data_dir=$1 # '/public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/260123-conplex_clusterretrain'
model_env_dir=$2 # '/public/home/lixy/..yanghao/646122/data_large/AOEDrug'

learning_rate=0.001
batch_size=8
epoch=2
dropout=0.1

# data processed
cd $model_env_dir/data_adapter
python predict_data_part_weightfind.py $data_dir 10000000 random
python predict_process_data_intergrated_new.py $data_dir random
processed_data_dir=$data_dir/data

# TransformerCPI retrain
echo "TransformerCPI retrain"
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate transformerCPI

cd $processed_data_dir/TransformerCPI
mkdir -p models results myouts
cd $model_env_dir/base_model/TransformerCPI/GPCR
python mol_featurizer_integ_retrain.py \
        $processed_data_dir \
        train \
        > $processed_data_dir/TransformerCPI/mol_featurizer.file 2>&1 & 
wait

export CUDA_VISIBLE_DEVICES=''
nohup python main_integ_trans.py \
        $processed_data_dir/TransformerCPI/data/train/ \
        $processed_data_dir/TransformerCPI/data/dev/ \
        -r $learning_rate \
        -e $epoch \
        -D $dropout \
        -b $batch_size \
        -m $processed_data_dir/TransformerCPI/models/model-$learning_rate-$batch_size-$dropout-trans.model \
        -o $processed_data_dir/TransformerCPI/results/validation_output_best_$batch_size-trans.csv > $processed_data_dir/TransformerCPI/myouts/myout_$learning_rate-$batch_size-$dropout-trans.file 2>&1 &     
wait

# ConPLex retrain
echo "ConPLex retrain"
conda activate conplex
cd $processed_data_dir/ConPLex
mkdir -p models results myouts
cd $model_env_dir/base_model/ConPLex_dev

export CUDA_VISIBLE_DEVICES=''
nohup python main_integ_conplex.py \
        --exp-id Conplex \
        $processed_data_dir/ConPLex/data \
        -r $learning_rate \
        -e $epoch \
        -D $dropout \
        -b $batch_size \
        -m $processed_data_dir/ConPLex/models/model-$learning_rate-$batch_size-$dropout-conplex.model \
        -o $processed_data_dir/ConPLex/results/validation_output-conplex.csv \
        -c configs/default_config.yaml \
        > $processed_data_dir/ConPLex/myouts/myout-$learning_rate-$batch_size-$dropout-conplex.file 2>&1 &
wait
# DeepConv-DTI retrain
echo "DeepConv retrain"
conda activate DeepConv-DTI
cd $processed_data_dir/DeepConv-DTI
mkdir -p models results myouts
cd $model_env_dir/base_model/DeepConv-DTI

export CUDA_VISIBLE_DEVICES=''
nohup python main_integ_DeepConvDTI.py \
        $processed_data_dir/DeepConv-DTI/data/train/dti.csv \
        $processed_data_dir/DeepConv-DTI/data/train/compound.csv \
        $processed_data_dir/DeepConv-DTI/data/train/protein.csv \
        --validation \
        -n $processed_data_dir \
        -i $processed_data_dir/DeepConv-DTI/data/dev/dti.csv \
        -d $processed_data_dir/DeepConv-DTI/data/dev/compound.csv \
        -t $processed_data_dir/DeepConv-DTI/data/dev/protein.csv \
        -W \
        -c 512 128 \
        -w 10 15 20 25 30 \
        -p 128 \
        -f 128 \
        -r $learning_rate \
        -n $processed_data_dir \
        -v Convolution \
        -l 2500 \
        -V morgan_fp_r2 \
        -L 2048 \
        -D $dropout \
        -a elu \
        -F 128 \
        -b $batch_size \
        -y 0.0001 \
        -o $processed_data_dir/DeepConv-DTI/results/validation_output_best_$batch_size-deepconv.csv \
        -m $processed_data_dir/DeepConv-DTI/models/model-$learning_rate-$batch_size-$dropout-deepconv \
        -e $epoch  \
        > $processed_data_dir/DeepConv-DTI/myouts/myout_$learning_rate-$batch_size-$dropout-deepconv.file 2>&1 &
wait
# DrugBAN retrain
echo "DrugBAN retrain"
conda activate drugban
cd $processed_data_dir/DrugBAN
mkdir -p models results myouts
cd $model_env_dir/base_model/DrugBAN

export CUDA_VISIBLE_DEVICES=''
nohup python main_integ_drugban.py $processed_data_dir/DrugBAN/data \
        -r $learning_rate \
        -e $epoch \
        -b $batch_size \
        -m $processed_data_dir/DrugBAN/models/model-$learning_rate-$batch_size-$dropout-drugban.model \
        -o $processed_data_dir/DrugBAN/results/validation_output-drugban.csv \
        --data Drugban \
        > $processed_data_dir/DrugBAN/myouts/$data_sub/myout-$learning_rate-$batch_size-$dropout-drugban.file 2>&1 & 
wait

