import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem
from collections import defaultdict
from functools import lru_cache
from multiprocessing import Pool, cpu_count
import numpy as np

# ---------- 基础函数 ----------
def mol_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol if mol is not None else None
    except:
        return None

def get_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

@lru_cache(maxsize=None)  # 缓存所有SMILES对应的指纹
def get_fp_from_smiles(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return get_fingerprint(mol)

def build_train_index(train_df, smiles_col, protein_col):
    train_index = defaultdict(list)
    for _, row in train_df.iterrows():
        smiles = row[smiles_col]
        protein = row[protein_col]
        fp = get_fp_from_smiles(smiles)  # 使用缓存
        if fp is not None:
            train_index[protein].append(fp)
    return train_index

def calculate_max_sim_for_one(args):
    """单个测试样本的处理函数，用于并行"""
    smiles, protein, train_index = args
    test_fp = get_fp_from_smiles(smiles)
    if test_fp is None:
        return 0.0, '<=0.5', '<=0.5'
    train_fps = train_index.get(protein, [])
    if not train_fps:
        return 0.0, '<=0.5', '<=0.5'
    sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
    max_sim = max(sims)
    # 分类
    if max_sim <= 0.5:
        fine = '<=0.5'
        coarse = '<=0.5'
    elif max_sim <= 0.6:
        fine = '0.5-0.6'
        coarse = '0.5-0.7'
    elif max_sim <= 0.7:
        fine = '0.6-0.7'
        coarse = '0.5-0.7'
    elif max_sim <= 0.8:
        fine = '0.7-0.8'
        coarse = '0.7-1'
    elif max_sim <= 0.9:
        fine = '0.8-0.9'
        coarse = '0.7-1'
    elif max_sim < 1.0:
        fine = '0.9-1'
        coarse = '0.7-1'
    else:
        fine = '1'
        coarse = '1'
    return max_sim, fine, coarse

def process_test_parallel(test_df, train_index, smiles_col, protein_col, n_workers=None):
    if n_workers is None:
        n_workers = cpu_count()  # 使用所有CPU核心
    # 准备参数列表
    args_list = [(row[smiles_col], row[protein_col], train_index) for _, row in test_df.iterrows()]
    with Pool(processes=n_workers) as pool:
        results = pool.map(calculate_max_sim_for_one, args_list)
    max_sims, fine_classes, coarse_classes = zip(*results) if results else ([], [], [])
    result_df = test_df.copy()
    result_df['Max_Similarity'] = max_sims
    result_df['Similarity_Class_Fine'] = fine_classes
    result_df['Similarity_Class_Coarse'] = coarse_classes
    return result_df

def main(test_csv, train_csv, output_csv,
         test_smiles_col='Compound_ID', test_protein_col='Protein_ID',
         train_smiles_col='Compound_ID', train_protein_col='Protein_ID',
         n_workers=None):
    print("读取训练集...")
    train_df = pd.read_csv(train_csv)
    print(f"训练集 {len(train_df)} 条")
    print("构建训练索引（指纹缓存）...")
    train_index = build_train_index(train_df, train_smiles_col, train_protein_col)
    print(f"共 {len(train_index)} 种蛋白")
    print("读取测试集...")
    test_df = pd.read_csv(test_csv)
    print(f"测试集 {len(test_df)} 条")
    print(f"使用 {n_workers or cpu_count()} 个进程并行计算...")
    result_df = process_test_parallel(test_df, train_index, test_smiles_col, test_protein_col, n_workers)
    print(f"保存至 {output_csv}")
    result_df.to_csv(output_csv, index=False)
    print("完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Classify each test compound by its max Tanimoto similarity '
                    '(vs. training compounds tested against the same protein)')
    parser.add_argument('--test_csv', type=str, required=True, help='Test set CSV path')
    parser.add_argument('--train_csv', type=str, required=True, help='Training set CSV path')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV path')
    parser.add_argument('--test_smiles_col', type=str, default='SMILES')
    parser.add_argument('--test_protein_col', type=str, default='Protein')
    parser.add_argument('--train_smiles_col', type=str, default='SMILES')
    parser.add_argument('--train_protein_col', type=str, default='Protein')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: all CPU cores)')
    args = parser.parse_args()

    main(
        test_csv=args.test_csv,
        train_csv=args.train_csv,
        output_csv=args.output_csv,
        test_smiles_col=args.test_smiles_col,
        test_protein_col=args.test_protein_col,
        train_smiles_col=args.train_smiles_col,
        train_protein_col=args.train_protein_col,
        n_workers=args.n_workers,
    )