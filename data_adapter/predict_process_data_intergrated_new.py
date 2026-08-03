import csv
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import sys
import os
import time
from itertools import combinations
from multiprocessing import Pool, cpu_count
# from tqdm import tqdm
# import hashlib
from functools import lru_cache

def data_file_dir_csv(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    return csv_files

def data_file_dir(folder_path):
    files = [file for file in os.listdir(folder_path)]
    return files 

def GetMorganFPs(mol, nBits=2048, radius=3, return_bitInfo=False):
    bitInfo = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, 
                                              bitInfo=bitInfo, nBits=nBits)
    arr = np.zeros((0,))
    np.set_printoptions(threshold=sys.maxsize)
    DataStructs.ConvertToNumpyArray(fp, arr)
    if return_bitInfo:
        return arr, bitInfo
    return arr

def optimized_morgan_to_string(arr):
    """优化Morgan指纹到字符串的转换"""
    return "\t".join(map(str, arr.astype(int)))

@lru_cache(maxsize=10000)
def get_cached_morgan_fp(smiles, nBits=2048, radius=2):
    """缓存Morgan指纹计算结果"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    
    arr = GetMorganFPs(mol, nBits=nBits, radius=radius, return_bitInfo=False)
    return optimized_morgan_to_string(arr)

def _process_single_smiles(args):
    """处理单个SMILES的辅助函数"""
    smiles, nBits, radius = args
    return get_cached_morgan_fp(smiles, nBits, radius)

def get_morgan_fp_parallel(smiles_list, nBits=2048, radius=2):
    """并行处理Morgan指纹生成"""
    # 对于小数据集，使用单进程可能更快
    if len(smiles_list) < 100:
        results = []
        for smiles in smiles_list:
            results.append(_process_single_smiles((smiles, nBits, radius)))
        return results
    
    # 对于大数据集使用并行处理
    with Pool(processes=min(cpu_count(), 8)) as pool:  # 限制最大进程数
        results = list(
            pool.imap(_process_single_smiles, [(smiles, nBits, radius) for smiles in smiles_list]))
    return results

def delete_data_smilesmol_none(csv_file, output_dir, file_name):
    # 读取 CSV 文件，指定 SMILES 列的数据类型为字符串
    df_used_data = pd.read_csv(csv_file, dtype={'SMILES': str})
    df_used_data.columns = ['SMILES', 'Sequence', 'Label']
    cols_to_dedup = ['SMILES', 'Sequence', 'Label']
    df_used_data = df_used_data[cols_to_dedup]
    # 过滤掉 NaN 值
    df_used_data = df_used_data.dropna(subset=['SMILES'])
    
    def is_valid_smiles(x):
        return isinstance(x, str) and Chem.MolFromSmiles(x) is not None
    
    mask = df_used_data['SMILES'].apply(is_valid_smiles)
    df_valid_data = df_used_data[mask]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    df_valid_data.to_csv(f'{output_dir}/{file_name}', index=False)

def optimized_shift_data_deepconv(csv_file, output_dir, file_name):
    print(f"Starting DeepConv-DTI conversion for {file_name}")
    start_time = time.time()
    
    df_used_data = pd.read_csv(csv_file)
    df_used_data.columns = ['Compound', 'Protein', 'Label']

    cols_to_dedup = ['Protein', 'Compound']
    dedup_cols = df_used_data[cols_to_dedup]

    # 1. 使用并行处理生成所有指纹
    print("Generating Morgan fingerprints...")
    unique_compounds = dedup_cols['Compound'].unique()
    
    # 并行生成指纹
    fp_results = get_morgan_fp_parallel(unique_compounds)
    
    # 创建化合物到指纹的映射
    compound_to_fp = dict(zip(unique_compounds, fp_results))

    # 2. 创建ID映射
    id_dfs = []
    for col in cols_to_dedup:
        _, uniques = pd.factorize(dedup_cols[col])
        prefix = 'Prot' if col == 'Protein' else 'Comp'
        id_df = pd.DataFrame({
            f'{col}_ID': [f'{prefix}{str(i+1).zfill(6)}' for i in range(len(uniques))], 
            col: uniques
        })
        id_dfs.append(id_df)

    df_Sequence_mapping = id_dfs[0]
    df_Smiles_mapping = id_dfs[1]

    # 3. 替换为ID
    for col in cols_to_dedup:
        mapping_dict = id_dfs[cols_to_dedup.index(col)].set_index(col)[f'{col}_ID'].to_dict()
        df_used_data[col] = df_used_data[col].map(mapping_dict)

    # # 4. 准备输出目录
    # if file_name == 'val.csv':
    #     file_name = 'dev.csv'  # 修正赋值操作
    
    output_subdir = f'{file_name[:-4]}'
    deepconv_output_dir = f'{output_dir}/DeepConv-DTI/data/{output_subdir}'
    os.makedirs(deepconv_output_dir, exist_ok=True)

    # 5. 保存蛋白质映射
    df_Sequence_mapping = df_Sequence_mapping.rename(columns={'Protein': 'Sequence'})
    df_Sequence_mapping.to_csv(
        f'{deepconv_output_dir}/protein.csv', 
        index=False, encoding='utf-8'
    )
    print(f'Protein mapping saved: {deepconv_output_dir}/protein.csv')

    # 6. 保存DTI数据
    df_used_data = df_used_data.rename(columns={'Protein': 'Protein_ID', 'Compound': 'Compound_ID'})
    df_used_data.to_csv(
        f'{deepconv_output_dir}/dti.csv', 
        index=False, encoding='utf-8'
    )
    print(f'DTI data saved: {deepconv_output_dir}/dti.csv')

    # 7. 保存化合物映射（使用预计算的指纹）
    df_Smiles_mapping = df_Smiles_mapping.rename(columns={'Compound': 'SMILES'})
    df_Smiles_mapping['morgan_fp_r2'] = df_Smiles_mapping['SMILES'].map(compound_to_fp)
    
    # 过滤掉无效的指纹
    valid_mask = df_Smiles_mapping['morgan_fp_r2'] != ""
    df_Smiles_mapping = df_Smiles_mapping[valid_mask]
    
    df_Smiles_mapping.to_csv(
        f'{deepconv_output_dir}/compound.csv', 
        index=False, encoding='utf-8'
    )
    print(f'Compound mapping saved: {deepconv_output_dir}/compound.csv')
    
    end_time = time.time()
    print(f"DeepConv-DTI conversion completed in {end_time - start_time:.2f} seconds")

def shift_data_drugban(csv_file, output_dir, file_name):
    df_used_data = pd.read_csv(csv_file)
    df_used_data.columns = ['SMILES', 'Protein', 'Y']
    drugban_output_dir = f'{output_dir}/DrugBAN/data'
    os.makedirs(drugban_output_dir, exist_ok=True)
    df_used_data.to_csv(f'{drugban_output_dir}/{file_name[:-4]}.csv', index=True, encoding='utf-8')
    print(f'DrugBAN data saved: {drugban_output_dir}/{file_name[:-4]}.csv')

def shift_data_conplex(csv_file, output_dir, file_name):
    df_used_data = pd.read_csv(csv_file)
    df_used_data.columns = ['SMILES', 'Target Sequence', 'Label']
    conplex_output_dir = f'{output_dir}/ConPLex/data/{file_name[:-4]}'
    os.makedirs(conplex_output_dir, exist_ok=True)
    df_used_data.to_csv(f'{conplex_output_dir}/test.csv', index=True, encoding='utf-8')
    print(f'ConPLex data saved: {conplex_output_dir}/test.csv')

def shift_data_transformercpi(csv_file, output_dir, file_name):
    df_used_data = pd.read_csv(csv_file)
    transformercpi_output_dir = f'{output_dir}/TransformerCPI/data'
    os.makedirs(transformercpi_output_dir, exist_ok=True)
    df_used_data.to_csv(f'{transformercpi_output_dir}/{file_name[:-4]}.txt', index=False, sep=' ', header=False)
    print(f'TransformerCPI data saved: {transformercpi_output_dir}/{file_name[:-4]}.txt')

def optimized_processing_pipeline():
    """优化的完整处理流程"""
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    mode = sys.argv[2]
    folder_path = f'{input_path}/part'
    # if mode=='predict':
    #     folder_path = f'{input_path}/part'
    # else:
    #     folder_path = f'{input_path}'
    
    # 检查输入目录是否存在
    if not os.path.exists(folder_path):
        print(f"Error: Input directory {folder_path} does not exist")
        sys.exit(1)
        
    os.makedirs(f'{input_path}/part_new', exist_ok=True)
    folder_path_new = f'{input_path}/part_new'
    output_dir = f'{input_path}/data'

    csv_files = data_file_dir_csv(folder_path)
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        csv_file_dir = f'{folder_path}/{csv_file}'
        print(f"\n{'='*50}")
        print(f"Processing: {csv_file_dir}")
        print(f"{'='*50}")

        # 1. 清理数据
        print("Step 1: Cleaning data...")
        delete_data_smilesmol_none(csv_file_dir, folder_path_new, file_name)
        csv_file_new = f'{folder_path_new}/{csv_file}'
        print(f"Cleaned data saved: {csv_file_new}")

        # 2. 转换格式（使用优化版本）
        print("Step 2: Converting to DeepConv-DTI format...")
        optimized_shift_data_deepconv(csv_file_new, output_dir, file_name)

        # 3. 其他格式转换
        print("Step 3: Converting to DrugBAN format...")
        shift_data_drugban(csv_file_new, output_dir, file_name)
        
        print("Step 4: Converting to ConPLex format...")
        shift_data_conplex(csv_file_new, output_dir, file_name)
        
        print("Step 5: Converting to TransformerCPI format...")
        shift_data_transformercpi(csv_file_new, output_dir, file_name)
        
        print(f"Completed processing {file_name}")

if __name__ == "__main__":
    total_start = time.time()
    optimized_processing_pipeline()
    total_end = time.time()
    print(f"\n{'='*50}")
    print(f"All processing completed in {total_end - total_start:.2f} seconds")
    print(f"{'='*50}")