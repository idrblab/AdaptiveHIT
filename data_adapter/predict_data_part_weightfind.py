import os
import sys
import pandas as pd

# python /public/home/lixy/..yanghao/646122/data_large/predict_data_part.py D:/.destbook/IDRB_AI/CPI/INtergratedCPI/cpi_data/all_DUDE 10000

input_path = sys.argv[1]
num_part = int(sys.argv[2])
mode = sys.argv[3]
output_path = f'{input_path}/part'
os.makedirs(output_path, exist_ok=True)

# 列名映射（支持两种格式）
column_mapping = {
    'SMILES': ['Compound_ID', 'SMILES', 'smiles', 'Smiles', 'compound', 'Compound', 'drug', 'Drug'],
    'Protein': ['Protein_ID', 'Protein', 'protein', 'sequence', 'Sequence', 'target', 'Target'],
    'Y': ['Y', 'label']
}

for filename in os.listdir(input_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_path, filename)
        df = pd.read_csv(file_path)
        
        # 自动识别并重命名列
        rename_dict = {}
        for target, candidates in column_mapping.items():
            for col in candidates:
                if col in df.columns:
                    rename_dict[col] = target
                    break
        
        # 重命名并选择需要的列
        df = df.rename(columns=rename_dict)[list(column_mapping.keys())]
        
        # 分割并保存
        for i, start in enumerate(range(0, len(df), num_part)):
            chunk_df = df.iloc[start:start + num_part]
            if mode=='predict':
                new_filename = f'{os.path.splitext(filename)[0]}_part{i+1}.csv'
            else:
                new_filename = f'{os.path.splitext(filename)[0]}.csv'
            chunk_df.to_csv(os.path.join(output_path, new_filename), index=False)