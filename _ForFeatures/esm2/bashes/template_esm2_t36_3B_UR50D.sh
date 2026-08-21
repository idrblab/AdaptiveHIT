#!/usr/bin/bash


# esm2type = 'esm2_t33_650M_UR50D'
# repr_layer = 33
# hid_dim = 1280
# esm2type = 'esm2_t36_3B_UR50D'
# repr_layer = 36
# hid_dim = 2560
# esm2type = 'esm2_t48_15B_UR50D'
# repr_layer = 48
# hid_dim = 5120
DATA_DIR=$1
DATATYPE=$2
cd $ADAP_MODEL_ROOT/_ForFeatures/esm2/bashes
python -u ../main_gpu.py --start_index head --esm2type esm2_t36_3B_UR50D --datatype ${DATATYPE} --repr_layer 36 --hid_dim 2560 --data_dir ${DATA_DIR}
