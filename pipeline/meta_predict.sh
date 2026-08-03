# #!/bin/bash
set -e
model_env_dir=$1

missions=("toy_dataset")
for mission in "${missions[@]}"; do
    data_dir=$model_env_dir/dataset/$mission

    model_dir_ConPLex="$model_env_dir/models/model-0.0005-64-0.1-conplex-858.model"
    DEVICES_ConPLex=''
    model_dir_DeepConv="$model_env_dir/models/model-0.0005-64-0.1-deep-969.model"
    DEVICES_DeepConv=''
    model_dir_DrugBAN="$model_env_dir/models/model-0.0005-64-0.1-drugban-940.model"
    DEVICES_DrugBAN=''
    model_dir_TransformerCPI="$model_env_dir/models/model-0.0005-128-0.1-trans-946.model"
    DEVICES_TransformerCPI=''

    source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
    conda activate drugban
    export CUDA_VISIBLE_DEVICES=2
    cd $model_env_dir/data_adapter

    # data processed
    cd $model_env_dir/data_adapter
    python predict_data_part_weightfind.py $data_dir 10000 random
    python predict_process_data_intergrated_new.py $data_dir random

    echo "data_process_id for embedding LLM !!!"
    python data_process_id.py $data_dir $mission

    # NOTE: embedding *generation* (X-Mol fine-tune inference, ESM-2 embedding
    # extraction) needs the full X-Mol/ESM-2 toolchains, which are NOT vendored in
    # this repo (only the trimmed weights needed to *use* precomputed embeddings
    # ship here, under data_adapter/xmol_weights/ -- see README). Obtain those
    # toolchains separately and place them at $model_env_dir/_ForFeatures/{xmol,esm2}
    # (matching the layout these scripts expect) to regenerate embeddings for a new
    # dataset; the shipped pretrained checkpoint (checkpoints/meta/) already has its
    # embeddings baked in and does not need this step.
    echo "xmol for mol !!!"
    conda activate xmol
    bash $model_env_dir/_ForFeatures/xmol/bashes/run_emb_all_new.sh $mission $data_dir 
    conda activate drugban
    python prebuild_xmol_cache.py \
            $data_dir \
            $mission \
            $model_env_dir/_ForFeatures/xmol/FT_to_embedding/data/for_output

    echo "esm2 for prots !!!"
    conda activate esm2
    bash $model_env_dir/_ForFeatures/esm2/bashes/template_esm2_t36_3B_UR50D.sh  $data_dir $mission

    cd $model_env_dir/pipeline/predict

    # DrugBAN predict
    echo "DrugBAN predict"
    conda activate drugban
    bash integ_screen_drugban_predict.sh \
            $model_dir_DrugBAN \
            $DEVICES_DrugBAN \
            $data_dir/data/DrugBAN \
            $model_env_dir/base_model/DrugBAN \
            > $data_dir/data/DrugBAN/DrugBAN.file 2>&1 &
    wait

    # DeepConv-DTI predict
    echo "DeepConv-DTI predict"
    conda activate DeepConv-DTI
    bash integ_screen_deep_predict.sh \
            $model_dir_DeepConv \
            $DEVICES_DeepConv \
            $data_dir/data/DeepConv-DTI \
            $model_env_dir/base_model/DeepConv-DTI \
            > $data_dir/data/DeepConv-DTI/DeepConv-DTI.file 2>&1 &
    wait

    # # ConPLex predict
    echo "ConPLex predict"
    conda activate conplex
    bash integ_screen_conplex_predict.sh \
            $model_dir_ConPLex \
            $DEVICES_ConPLex \
            $data_dir/data/ConPLex \
            $model_env_dir/base_model/ConPLex_dev \
            > $data_dir/data/ConPLex/ConPLex.file 2>&1 &
    wait

    # TransformerCPI predict
    echo "transformerCPI predict"
    conda activate transformerCPI
    bash integ_screen_trans_predict.sh \
                $model_dir_TransformerCPI \
                $DEVICES_TransformerCPI \
                $data_dir/data/TransformerCPI \
                $model_env_dir/base_model/TransformerCPI \
                > $data_dir/data/TransformerCPI/TransformerCPI.file 2>&1 &
    wait

    cd $model_env_dir/data_adapter
    # process_result_data
    echo "process_result_data !!!"
    python result_process_data_integ_and_evalu.py $data_dir/data predict
    wait

    # # python label_ori_merge.py $data_dir predict $mission
    echo "label_ori_merge !!!"
    python label_ori_merge.py $data_dir predict $mission
    wait
done


export CUDA_VISIBLE_DEVICES=''
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate drugban
conda env list

cd $model_env_dir

missions_list=("toy_dataset")
for mission in "${missions_list[@]}"; do
    data_dir=$model_env_dir/dataset/$mission

    cd $model_env_dir/meta_learner
    python predict.py \
        --input_dir $data_dir/data/end_merged \
        --model_dir $data_dir/models_meta \
        --weights_dir $data_dir/weights  \
        --output_dir $data_dir/results_meta/$mission \
        --dataset_name $mission \
        --mode 4\
        --models_mode all \
        --base_models TransformerCPI DeepConv-DTI ConPLex DrugBAN \
        --strategies \
            average \
            vote-all-1 \
            weighted_logistic_balanced \
            meta_full_esm2_xmol_prob_attention-32-0.001 \
        --eval \
        --eval_modes all diff similarity diff_similarity  \
        --eval_base_models \
        --data_subdir $mission
done