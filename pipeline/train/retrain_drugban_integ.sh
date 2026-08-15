#!/bin/bash
# nohup bash /public/home/lixy/..yanghao/646122/DrugBAN-main/retrain_drugban_integ.sh /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/DATA_bindingdb_similarity/DrugBAN 4 0.001 128 > /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/DATA_bindingdb_similarity/DrugBAN/retrain_drugban_integ.file 2>&1 &
# "0.001" "0.005" "0.0001" "0.0005"
data_dir=$1 # "biosnap"
DEVICES=$2 # 0
learning_rate=$3 # "var1,var2,var3"
batch_size=$4
## 将变量列表转换为数组
# IFS=',' read -ra learning_rate_values <<< "$learning_rates"
dropout=0.1
epoch=$5   # 参数 B 的可选值
# data_dir_snbs=("random_6" "random_7" "random_8" "random_9" "random_10")
# data_dir_snbs=("random_1" "random_2" "random_3" "random_4" "random_5")
data_dir_snbs=("random_2" "random_3")
model_env_dir=$6  # path to the DrugBAN submodule (base_model/DrugBAN)

source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate drugban-A100
conda env list

# echo "data_dir: $data_dir"
# echo "learning-rate: $learning_rate"
# echo "batch-size: $batch_size"
# echo "dropout: $dropout"
# echo "----------------"

for data_sub in "${data_dir_snbs[@]}"; do

    echo "data_dir: $data_dir/data/$data_sub"
    echo "learning-rate: $learning_rate"
    echo "batch-size: $batch_size"
    echo "dropout: $dropout"
    echo "----------------"
    if [ -e "$data_dir/models/$data_sub/models/model-$learning_rate-$batch_size-$dropout-drugban_best1.model" ]; then
        echo "model file exists,skipping the command"
    else
        echo "File does not exist, executing the command"
        # export CUDA_VISIBLE_DEVICES=$DEVICES
        export CUDA_VISIBLE_DEVICES=$DEVICES
        python $model_env_dir/main_integ_drugban.py $data_dir/data/$data_sub -r $learning_rate -e $epoch -b $batch_size -m $data_dir/models/$data_sub/model-$learning_rate-$batch_size-$dropout-drugban.model -o $data_dir/results/$data_sub/validation_output.csv --data Drugban > $data_dir/myouts/$data_sub/myout-$learning_rate-$batch_size-$dropout.file 2>&1 & 
        wait
    fi
done

# export CUDA_VISIBLE_DEVICES=''
# # python main_denpy.py $data_dir/train/ $data_dir/dev/ -r $learning_rate -e 50 -D $dropout -b $batch_size -m $data_dir/models/model-$learning_rate-$batch_size-$dropout.model -o $data_dir/results/validation_output_best_$batch_size.csv > $data_dir/myouts/myout_-$learning_rate-$batch_size-$dropout.file 2>&1 &     
# nohup python /public/home/lixy/..yanghao/646122/DrugBAN-main/main_new.py $data_dir -r $learning_rate -e 50 -b $batch_size -m $data_dir/models/model-$learning_rate-$batch_size-$dropout.model -o $data_dir/results/validation_output.csv --data Drugban > $data_dir/myouts/myout-$learning_rate-$batch_size-$dropout.file 2>&1 & 
# wait

#nohup bash /public/home/lixy/..yanghao/646122/DrugBAN-main/retrain_drugban.sh /public/home/lixy/..yanghao/646122/data_large/JPA_hdac_sub/DrugBAN 0 0.0005 64 > /public/home/lixy/..yanghao/646122/data_large/JPA_hdac_sub/myout_retrain.file 2>&1 &

# nohup python main_new.py ./drugban_cpi_data/$data_dir/$data_sub -r $learning_rate -e 50 -b $batch_size -m ./output/$data_dir/models/$data_sub/model_$data_dir-$learning_rate-$batch_size-$dropout.model -o ./output/$data_dir/results/$data_sub/validation_output__$data_dir.csv --data $data_dir-$data_sub > ./output/$data_dir/myouts/$data_sub/myout_$data_dir-$learning_rate-$batch_size-$dropout.file 2>&1 &