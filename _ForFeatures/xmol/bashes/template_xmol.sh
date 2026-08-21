#!/usr/bin/bash
set -euo pipefail
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate iTarget


cd ../FT_to_embedding/data/for_output/
cp pre_process.py pre_process.runtmp.py
sed -i "s|datatype = 'template'|datatype = 'self'|g" pre_process.runtmp.py
python pre_process.runtmp.py
# sed -i "s|datatype = 'self'|datatype = 'template'|g" pre_process.py
cd ../../../


cd ./FT_to_embedding
cp ./script/run_emb.sh ./script/run_emb.runtmp.sh
sed -i "s|--train_set \${TASK_DATA_PATH}/template|--train_set \${TASK_DATA_PATH}/self|g" ./script/run_emb.runtmp.sh
sed -i "s|--test_set \${TASK_DATA_PATH}/template|--test_set \${TASK_DATA_PATH}/self|g" ./script/run_emb.runtmp.sh
echo "XMOL is Running......May Take Hours......Please Wait......Logs are located in ./_ForFeatures/xmol/FT_to_embedding/log/launch.log"
bash +x ./script/run_emb.runtmp.sh 2>&1 | grep -v '^+'
# sed -i "s|--train_set \${TASK_DATA_PATH}/self|--train_set \${TASK_DATA_PATH}/template|g" ./script/run_emb.sh
# sed -i "s|--test_set \${TASK_DATA_PATH}/self|--test_set \${TASK_DATA_PATH}/template|g" ./script/run_emb.sh


cd ../FT_to_embedding/data/for_output/
cp post_process.py post_process.runtmp.py
sed -i "s|datatype = 'template'|datatype = 'self'|g" post_process.runtmp.py
python post_process.runtmp.py
# sed -i "s|datatype = 'self'|datatype = 'template'|g" post_process.py
cd ../../../

cd ./bashes
