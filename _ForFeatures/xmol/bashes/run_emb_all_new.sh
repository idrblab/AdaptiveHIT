#!/usr/bin/env bash
set -eux

# ========== Configuration ==========
DATATYPE=$1
INPUT_DIR=$2

# X-Mol needs PaddlePaddle 1.8.5, which cannot coexist with the adaptivehit
# env's torch stack -- run.sh builds a dedicated `xmol` env for it. This
# script is invoked with `bash`, so activating here does not leak back to
# the caller.
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate xmol

cd ${ADAP_MODEL_ROOT}/_ForFeatures/xmol/FT_to_embedding/data/for_output
python pre_process_new.py ${INPUT_DIR}/id ${DATATYPE}

# Set environment paths
cd ${ADAP_MODEL_ROOT}/_ForFeatures/xmol/FT_to_embedding
source ./slurm/env.sh
source ./slurm/utils.sh
source ./conf_pre/ft_conf.sh

# Environment optimization
export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.9   # GPU memory usage ratio
export CPU_NUM=8                                 # Number of CPU threads

# Batch processing parameters
BATCH_SIZE=4                # Batch size
LOAD_BATCH_SIZE=100000      # Preload batch size (molecules)
USE_MULTI_GPU=false         # Whether to use multi-GPU

# Check configuration
check_iplist

distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP}"

# run.sh installs a CPU-only paddlepaddle (no 1.8.x GPU build supports
# Ampere), so default to CPU. Set XMOL_USE_CUDA=true only if you installed a
# matching paddlepaddle-gpu; CUDA_VISIBLE_DEVICES is then honoured as set by
# the caller rather than pinned to a fixed device here.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Run the optimized classifier
python -u ./run_classifier_new_end.py \
    --use_cuda "${XMOL_USE_CUDA:-false}" \
    --is_distributed false \
    --use_fast_executor ${e_executor:-"true"} \
    --tokenizer ${TOKENIZER:-"FullTokenizer"} \
    --use_fp16 ${use_fp16:-"false"} \
    --use_dynamic_loss_scaling ${use_fp16:-"false"} \
    --init_loss_scaling ${loss_scaling:-128} \
    --do_train false \
    --do_val false \
    --do_test false \
    --verbose true \
    --batch_size ${BATCH_SIZE} \
    --in_tokens false \
    --stream_job ${STREAM_JOB:-""} \
    --init_pretraining_params ${MODEL_PATH:-""} \
    --init_checkpoint ${CKPT_PATH:-""} \
    --train_set ${TASK_DATA_PATH}/${DATATYPE} \
    --test_set ${TASK_DATA_PATH}/${DATATYPE} \
    --vocab_path config/vocab.txt \
    --ernie_config_path config/ernie_config.json \
    --checkpoints ./checkpoints \
    --save_steps ${SAVE_STEPS} \
    --weight_decay 0.01 \
    --warmup_proportion ${WARMUP_PROPORTION:-"0.0"} \
    --validation_steps ${VALID_STEPS} \
    --epoch ${EPOCH} \
    --max_seq_len 256 \
    --learning_rate ${LR_RATE:-"1e-4"} \
    --skip_steps 10 \
    --num_iteration_per_drop_scope 1 \
    --num_labels ${num_labels} \
    --random_seed 1