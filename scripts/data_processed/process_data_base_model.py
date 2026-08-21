#!/usr/bin/env python3
"""
Unified data processing pipeline for DTI prediction.
Usage:
    python process_data.py <data_dir> <num_parts> <mode>
Example:
    python process_data.py /path/to/data 50000 predict
"""

import os
import sys
import time
import csv
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from functools import lru_cache
from multiprocessing import Pool, cpu_count


# --------------------- Part 1: Data Splitting ---------------------

def split_csv_files(input_path, num_part, mode):
    """
    Split each CSV file in input_path into chunks of `num_part` rows.
    Save chunks to <input_path>/part/ with naming:
        - if mode == 'predict': <original>_part{i+1}.csv
        - otherwise: <original>.csv (overwrites original, not recommended)
    """
    output_path = os.path.join(input_path, 'part')
    os.makedirs(output_path, exist_ok=True)

    column_mapping = {
        'SMILES': ['Compound_ID', 'SMILES', 'smiles', 'Smiles', 'compound', 'Compound', 'drug', 'Drug'],
        'Protein': ['Protein_ID', 'Protein', 'protein', 'sequence', 'Sequence', 'target', 'Target'],
        'Y': ['Y', 'label']
    }

    for filename in os.listdir(input_path):
        if not filename.endswith('.csv'):
            continue

        file_path = os.path.join(input_path, filename)
        df = pd.read_csv(file_path)

        # Auto-detect and rename columns
        rename_dict = {}
        for target, candidates in column_mapping.items():
            for col in candidates:
                if col in df.columns:
                    rename_dict[col] = target
                    break

        df = df.rename(columns=rename_dict)[list(column_mapping.keys())]

        # Split and save
        for i, start in enumerate(range(0, len(df), num_part)):
            chunk = df.iloc[start:start + num_part]
            if mode == 'predict':
                new_name = f'{os.path.splitext(filename)[0]}_part{i+1}.csv'
            else:
                new_name = f'{os.path.splitext(filename)[0]}.csv'
            chunk.to_csv(os.path.join(output_path, new_name), index=False)


# --------------------- Part 2: Format Conversion ---------------------

def get_morgan_fp(mol, nBits=2048, radius=3, return_bitInfo=False):
    """Compute Morgan fingerprint as numpy array."""
    bitInfo = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                               bitInfo=bitInfo, nBits=nBits)
    arr = np.zeros((0,))
    np.set_printoptions(threshold=sys.maxsize)
    DataStructs.ConvertToNumpyArray(fp, arr)
    if return_bitInfo:
        return arr, bitInfo
    return arr


def fp_to_string(arr):
    """Convert fingerprint array to tab-separated string."""
    return "\t".join(map(str, arr.astype(int)))


@lru_cache(maxsize=10000)
def cached_morgan_fp(smiles, nBits=2048, radius=2):
    """Cached computation of Morgan fingerprint from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    arr = get_morgan_fp(mol, nBits=nBits, radius=radius)
    return fp_to_string(arr)


def _process_single_smiles(args):
    """Helper for parallel SMILES processing."""
    smiles, nBits, radius = args
    return cached_morgan_fp(smiles, nBits, radius)


def parallel_morgan_fp(smiles_list, nBits=2048, radius=2):
    """Generate Morgan fingerprints in parallel."""
    if len(smiles_list) < 100:
        return [_process_single_smiles((s, nBits, radius)) for s in smiles_list]

    with Pool(processes=min(cpu_count(), 8)) as pool:
        return list(pool.imap(_process_single_smiles,
                              [(s, nBits, radius) for s in smiles_list]))


def clean_invalid_smiles(csv_file, output_dir, file_name):
    """Remove rows with invalid SMILES and save to output_dir."""
    df = pd.read_csv(csv_file, dtype={'SMILES': str})
    df.columns = ['SMILES', 'Sequence', 'Label']
    df = df[['SMILES', 'Sequence', 'Label']].dropna(subset=['SMILES'])

    def is_valid_smiles(x):
        return isinstance(x, str) and Chem.MolFromSmiles(x) is not None

    valid = df[df['SMILES'].apply(is_valid_smiles)]
    os.makedirs(output_dir, exist_ok=True)
    valid.to_csv(os.path.join(output_dir, file_name), index=False)


def convert_to_deepconv_dti(csv_file, output_dir, file_name):
    """Convert to DeepConv-DTI format (protein.csv, compound.csv, dti.csv)."""
    print(f"DeepConv-DTI conversion: {file_name}")
    start = time.time()

    df = pd.read_csv(csv_file)
    df.columns = ['Compound', 'Protein', 'Label']

    # Build unique mappings
    unique_compounds = df['Compound'].unique()
    compound_fps = parallel_morgan_fp(unique_compounds)
    compound_to_fp = dict(zip(unique_compounds, compound_fps))

    # Create ID mappings
    prot_ids, prot_uniques = pd.factorize(df['Protein'])
    comp_ids, comp_uniques = pd.factorize(df['Compound'])

    prot_df = pd.DataFrame({
        'Protein_ID': [f'Prot{str(i+1).zfill(6)}' for i in range(len(prot_uniques))],
        'Sequence': prot_uniques
    })
    comp_df = pd.DataFrame({
        'Compound_ID': [f'Comp{str(i+1).zfill(6)}' for i in range(len(comp_uniques))],
        'SMILES': comp_uniques
    })
    comp_df['morgan_fp_r2'] = comp_df['SMILES'].map(compound_to_fp)
    comp_df = comp_df[comp_df['morgan_fp_r2'] != ""]  # drop invalid

    # Map IDs to main dataframe
    prot_map = dict(zip(prot_uniques, prot_df['Protein_ID']))
    comp_map = dict(zip(comp_uniques, comp_df['Compound_ID']))
    df['Protein'] = df['Protein'].map(prot_map)
    df['Compound'] = df['Compound'].map(comp_map)
    df = df.rename(columns={'Protein': 'Protein_ID', 'Compound': 'Compound_ID'})

    # Save outputs
    subdir = os.path.join(output_dir, 'DeepConv-DTI', 'data', os.path.splitext(file_name)[0])
    os.makedirs(subdir, exist_ok=True)

    prot_df.to_csv(os.path.join(subdir, 'protein.csv'), index=False, encoding='utf-8')
    comp_df.to_csv(os.path.join(subdir, 'compound.csv'), index=False, encoding='utf-8')
    df.to_csv(os.path.join(subdir, 'dti.csv'), index=False, encoding='utf-8')

    print(f"DeepConv-DTI done in {time.time()-start:.2f}s")


def convert_to_drugban(csv_file, output_dir, file_name):
    """Convert to DrugBAN format (single CSV with headers)."""
    df = pd.read_csv(csv_file)
    df.columns = ['SMILES', 'Protein', 'Y']
    out_dir = os.path.join(output_dir, 'DrugBAN', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + '.csv')
    df.to_csv(out_path, index=True, encoding='utf-8')
    print(f"DrugBAN saved: {out_path}")


def convert_to_conplex(csv_file, output_dir, file_name):
    """Convert to ConPLex format (single test.csv)."""
    df = pd.read_csv(csv_file)
    df.columns = ['SMILES', 'Target Sequence', 'Label']
    out_dir = os.path.join(output_dir, 'ConPLex', 'data', os.path.splitext(file_name)[0])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'test.csv')
    df.to_csv(out_path, index=True, encoding='utf-8')
    print(f"ConPLex saved: {out_path}")


def convert_to_transformercpi(csv_file, output_dir, file_name):
    """Convert to TransformerCPI format (space-separated, no header)."""
    df = pd.read_csv(csv_file)
    df.columns = ['SMILES', 'Protein', 'Y']
    out_dir = os.path.join(output_dir, 'TransformerCPI', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + '.txt')
    df.to_csv(out_path, index=False, sep=' ', header=False, encoding='utf-8')
    print(f"TransformerCPI saved: {out_path}")


def process_all_parts(input_path, mode):
    """
    Process all CSV files in <input_path>/part:
        1. Clean invalid SMILES -> <input_path>/part_new
        2. Convert to all target formats -> <input_path>/data
    """
    part_dir = os.path.join(input_path, 'part')
    if not os.path.exists(part_dir):
        print(f"Error: {part_dir} does not exist. Run splitting first.")
        sys.exit(1)

    clean_dir = os.path.join(input_path, 'part_new')
    output_base = os.path.join(input_path, 'data')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(output_base, exist_ok=True)

    csv_files = [f for f in os.listdir(part_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {part_dir}")
        return

    print(f"Found {len(csv_files)} part files to process.")

    for csv_file in csv_files:
        file_path = os.path.join(part_dir, csv_file)
        print(f"\n--- Processing: {csv_file} ---")

        # Step 1: clean
        clean_file = os.path.join(clean_dir, csv_file)
        clean_invalid_smiles(file_path, clean_dir, csv_file)

        # Step 2: convert to all formats
        convert_to_deepconv_dti(clean_file, output_base, csv_file)
        convert_to_drugban(clean_file, output_base, csv_file)
        convert_to_conplex(clean_file, output_base, csv_file)
        convert_to_transformercpi(clean_file, output_base, csv_file)


# --------------------- Main Entry ---------------------

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    data_dir = sys.argv[1]
    num_parts = int(sys.argv[2])
    mode = sys.argv[3]

    print("=== Step 1: Splitting CSV files ===")
    split_csv_files(data_dir, num_parts, mode)

    print("\n=== Step 2: Converting to model formats ===")
    process_all_parts(data_dir, mode)

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    total_start = time.time()
    main()
    print(f"Total runtime: {time.time() - total_start:.2f} seconds")