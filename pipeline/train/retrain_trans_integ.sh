#!/bin/bash
# nohup bash mol_featurizer_trans_integ.sh /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/DATA_bindingdb_similarity/TransformerCPI 0.0001 32 > retrain_trans_integ.file  2>&1 & 
# "0.001" "0.005" "0.0001" "0.0005"
data_dir=$1 # "biosnap"
DEVICES=$2 # 0
learning_rate=$3 # "var1,var2,var3"
batch_size=$4
epoch=$5
## 将变量列表转换为数组
# IFS=',' read -ra learning_rate_values <<< "$learning_rates"
# data_dir_snbs=("random_1" "random_2" "random_3" "random_4" "random_5")
# data_dir_snbs=("random_2" "random_3" "random_4" "random_5")
# data_dir_snbs=("random_1")
data_dir_snbs=("random_6" "random_7" "random_8" "random_9" "random_10")

model_env_dir=$6  # path to the TransformerCPI submodule's GPCR dir (base_model/TransformerCPI/GPCR)

dropout=0.1   # 参数 B 的可选值

source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate transformerCPI
conda env list

for data_sub in "${data_dir_snbs[@]}"; do

    echo "data_dir: $data_dir"
    echo "learning-rate: $learning_rate"
    echo "batch-size: $batch_size"
    echo "dropout: $dropout"
    echo "----------------"
    if [ -e "$data_dir/data/$data_sub/models/model-$learning_rate-$batch_size-$dropout-trans_best1.model" ]; then
        echo "model file exists,skipping the command"
    else
        echo "File does not exist, executing the command"
        export CUDA_VISIBLE_DEVICES=$DEVICES
        nohup python $model_env_dir/main_integ_trans.py $data_dir/data/$data_sub/train/ $data_dir/data/$data_sub/dev/ -r $learning_rate -e $epoch -D $dropout -b $batch_size -m $data_dir/models/$data_sub/model-$learning_rate-$batch_size-$dropout-trans.model -o $data_dir/results/$data_sub/validation_output_best_$batch_size.csv > $data_dir/myouts/$data_sub/myout_$learning_rate-$batch_size-$dropout.file 2>&1 &     
        wait
    fi
done


# python main_integ_trans.py /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/TransformerCPI/data/random_1/train/ /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/TransformerCPI/data/random_1/dev/ -r 0.001 -e 1 -D 0.1 -b 512 -m /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/TransformerCPI/models/random_1/model-0.01-512-try-trans.model -o /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/TransformerCPI/results/random_1/validation_output_best_512.csv