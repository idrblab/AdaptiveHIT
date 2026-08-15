#!/bin/bash
# nohup bash /public/home/lixy/..yanghao/646122/DeepConv-DTI/retrain_deep_integ.sh /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/DATA_bindingdb_similarity/DeepConv-DTI 2 0.0001 32 > /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/DATA_bindingdb_similarity/DeepConv-DTI/retrain_deep_integ.file 2>&1 &
# "0.001" "0.005" "0.0001" "0.0005"
data_dir=$1 # "biosnap"
DEVICES=$2 # 0
learning_rate=$3 # "var1,var2,var3"
batch_size=$4
dropout=0.1
epoch=$5   # 参数 B 的可选值
## 将变量列表转换为数组
# IFS=',' read -ra learning_rate_values <<< "$learning_rates"
data_dir_snbs=("random_6" "random_7" "random_8" "random_9" "random_10")
# data_dir_snbs=("random_1" "random_2" "random_3" "random_4" "random_5")
# data_dir_snbs=("random_2" "random_3" "random_4" "random_5")
# data_dir_snbs=("random_1")
model_env_dir=$6  # path to the DeepConv-DTI submodule (base_model/DeepConv-DTI)

# data_subs=("random_1" "random_2" "random_3" "random_4" "random_5")  # 参数 B 的可选值
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate DeepConv-DTI
conda env list

for data_sub in "${data_dir_snbs[@]}"; do

    echo "data_dir: $data_dir/data/$data_sub"
    echo "learning-rate: $learning_rate"
    echo "batch-size: $batch_size"
    echo "dropout: $dropout"
    echo "----------------"
    if [ -e "$data_dir/models/$data_sub/models/model-$learning_rate-$batch_size-$dropout-deepconv_best1.model" ]; then
        echo "model file exists,skipping the command"
    else
        echo "File does not exist, executing the command"
        export CUDA_VISIBLE_DEVICES=$DEVICES
        nohup python $model_env_dir/main_integ_DeepConvDTI.py $data_dir/data/$data_sub/train/dti.csv $data_dir/data/$data_sub/train/compound.csv $data_dir/data/$data_sub/train/protein.csv --validation -n $data_sub -i $data_dir/data/$data_sub/val/dti.csv -d $data_dir/data/$data_sub/val/compound.csv -t $data_dir/data/$data_sub/val/protein.csv -W -c 512 128 -w 10 15 20 25 30 -p 128 -f 128 -r $learning_rate -n $data_sub -v Convolution -l 2500 -V morgan_fp_r2 -L 2048 -D $dropout -a elu -F 128 -b $batch_size -y 0.0001 -o $data_dir/results/$data_sub/validation_output_best__$batch_size.csv -m $data_dir/models/$data_sub/model-$learning_rate-$batch_size-$dropout-deepconv.model -e $epoch  > $data_dir/myouts/$data_sub/myout_-$learning_rate-$batch_size-$dropout.file 2>&1 &
        wait
    fi
done


# python main_integ_DeepConvDTI.py /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/DeepConv-DTI/data/random_1/train/dti.csv /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/DeepConv-DTI/data/random_1/train/compound.csv /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/DeepConv-DTI/data/random_1/train/protein.csv --validation -n random_1 -i /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/DeepConv-DTI/data/random_1/val/dti.csv -d /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/DeepConv-DTI/data/random_1/val/compound.csv -t /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/DeepConv-DTI/data/random_1/val/protein.csv -W -c 512 128 -w 10 15 20 25 30 -p 128 -f 128 -r 0.0001 -n random_1 -v Convolution -l 2500 -V morgan_fp_r2 -L 2048 -D 0.1 -a elu -F 128 -b 16 -y 0.0001 -o /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/DeepConv-DTI/results/random_1/validation_output_best__16.csv -m /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/250907_DATA_Itarget_retrain/DeepConv-DTI/models/random_1/model-$learning_rate-16-0.1-deepconv.model -e 1 &



