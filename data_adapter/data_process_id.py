import pandas as pd
import os
from pathlib import Path
import hashlib
import sys
# def extract_data_from_csv(file_path):
#     """从CSV文件中提取SMILES和序列数据"""
#     df = pd.read_csv(file_path)
    
#     # 自动检测列名（不区分大小写）
#     smiles_col = 'Compound_ID'
#     seq_col = 'Protein_ID'
    
#     if smiles_col is None or seq_col is None:
#         print(f"跳过 {file_path}: 未找到SMILES或序列列")
#         return [], []
    
#     smiles_data = df[smiles_col].dropna().astype(str).str.strip().tolist()
#     seq_data = df[seq_col].dropna().astype(str).str.strip().tolist()
    
#     return smiles_data, seq_data


def extract_data_from_csv(file_path):
    """从CSV文件中提取SMILES和序列数据"""
    df = pd.read_csv(file_path)
    
    # 可能的SMILES列名
    possible_smiles_cols = ['Compound_ID', 'SMILES', 'smiles', 'Smiles', 'compound', 'Compound', 'drug', 'Drug']
    # 可能的序列列名
    possible_seq_cols = ['Protein_ID', 'Protein', 'protein', 'sequence', 'Sequence', 'target', 'Target']
    
    # 自动检测SMILES列
    smiles_col = None
    for col in possible_smiles_cols:
        if col in df.columns:
            smiles_col = col
            break
    
    # 自动检测序列列
    seq_col = None
    for col in possible_seq_cols:
        if col in df.columns:
            seq_col = col
            break
    
    if smiles_col is None or seq_col is None:
        print(f"跳过 {file_path}: 未找到SMILES或序列列")
        print(f"  可用的列: {list(df.columns)}")
        return [], []
    
    smiles_data = df[smiles_col].dropna().astype(str).str.strip().tolist()
    seq_data = df[seq_col].dropna().astype(str).str.strip().tolist()
    
    return smiles_data, seq_data




def generate_id(item, prefix):
    """生成唯一的ID"""
    return f"{prefix}{hashlib.md5(item.encode()).hexdigest()[:8]}"

def process_csv_folder(folder_path, dataset_name):
    csv_files_all = []
    """处理文件夹中所有CSV文件"""

    print(dataset_name)
    if dataset_name in ['human', 'human_cold']:
        # folder_path_new = fr'{folder_path}/{dataset_name}/random_1/data/end_em'
        folder_path_new = fr'{folder_path}/random_1'
        folder = Path(folder_path_new)
        csv_files = list(folder.glob("*.csv"))
        csv_files_all.extend(csv_files)
    elif dataset_name in ['mutisimi']:
        random_list = ['random_1', 'random_2', 'random_3', 'random_4', 'random_5']
        for random_n in random_list:
            folder_path_new = fr'{folder_path}/{random_n}'
            folder = Path(folder_path_new)
            csv_files = list(folder.glob("*.csv"))
            csv_files_all.extend(csv_files)    
    else:
        # folder_path_new = fr'{folder_path}/{dataset_name}/data/end_em'
        folder_path_new = fr'{folder_path}'
        folder = Path(folder_path_new)
        csv_files = list(folder.glob("*.csv"))
        csv_files_all.extend(csv_files)




    if not csv_files:
        print("未找到CSV文件")
        return
    
    # 收集所有数据
    all_smiles = set()
    all_sequences = set()
    
    for csv_file in csv_files_all:
        smiles_list, seq_list = extract_data_from_csv(csv_file)
        all_smiles.update(smiles_list)
        all_sequences.update(seq_list)
        print(f"处理: {csv_file.name} -> {len(smiles_list)}个SMILES, {len(seq_list)}个序列")
    
    # 生成ID并创建DataFrame
    compounds_df = pd.DataFrame([
        {"drugid": generate_id(smiles, "DR"), "smiles": smiles}
        for smiles in all_smiles
    ])
    
    proteins_df = pd.DataFrame([
        {"protid": generate_id(seq, "PR"), "sequence": seq}
        for seq in all_sequences
    ])
    
    # 保存到CSV
    output_dir = fr'{folder_path}/id'
    os.makedirs(output_dir, exist_ok=True)
    
    compounds_df.to_csv(f"{output_dir}/{dataset_name}_drugs.csv", index=False)
    proteins_df.to_csv(f"{output_dir}/{dataset_name}_prots.csv", index=False)
    
    print(f"\n处理完成！")
    print(f"唯一化合物数: {len(compounds_df)}")
    print(f"唯一蛋白质数: {len(proteins_df)}")
    print(f"结果已保存到 {output_dir}/ 目录")

if __name__ == "__main__":
    # folder_path = '/public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/251128_weight_find/data'
    # dataset_list = ['human', 'human_cold', 'mutisimi', 'bindingdb', 'chembl_time']
    data_dir = sys.argv[1]
    dataset_name = sys.argv[2]
    
    folder_path = data_dir
    process_csv_folder(folder_path, dataset_name)