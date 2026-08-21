#!/bin/bash
model_dir=$1 # "/public/home/lixy/..yanghao/646122/data_large/model_best/model-0.0005-64-0.1-drugban-940.model"
DEVICES=$2 # 0
testset_dir=$3 
base_model_dir=$4 # "/public/home/lixy/..yanghao/646122/data_large/DATA_ADMET/DrugBAN"

cd $base_model_dir/GPCR
mkdir -p $testset_dir/results_adaptivehit   $testset_dir/myouts 
testset_csv_files=$(find "$testset_dir/data" -mindepth 1 -maxdepth 1 -type f -name "*.txt")

for data_sub in $testset_csv_files; do
    data_sub=("$(basename "$data_sub")")
    data_sub_name=("$(basename "$data_sub" .txt)")
    echo "data_dir: $testset_dir/data/$data_sub"
    echo "data_sub: $data_sub"
    echo "model_dir: $model_dir"
    echo "----------------"

    export CUDA_VISIBLE_DEVICES=$DEVICES
    nohup python predict_trans.py \
            $model_dir \
            predict \
            $data_sub_name \
            $testset_dir/data/$data_sub \
            -o $testset_dir/results_adaptivehit/ \
            > $testset_dir/myouts/$data_sub_name.file 2>&1
    wait
done