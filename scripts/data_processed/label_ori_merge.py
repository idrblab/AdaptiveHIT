import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy


def data_file_dir_csv(folder_path):
    """
    Return a list of CSV file names (without extension) from the given folder.
    """
    return [file[:-4] for file in os.listdir(folder_path) if file.endswith('.csv')]


def concat_csv_by_prefix(folder_path, file_prefix, output_file, mode, models_mode):
    """
    Concatenate all CSV files that match a given prefix and mode pattern.
    The pattern is: {file_prefix}*{mode}_{models_mode[-1]}.csv
    """
    csv_files = list(Path(folder_path).glob(f"{file_prefix}*{mode}_{models_mode[-1]}.csv"))

    if not csv_files:
        return pd.DataFrame()

    concated_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    return concated_df


def label_csv_merge(concated_df, df_ori, file_prefix, output_file):
    """
    Merge the concatenated predictions with original label data.
    Automatically detects column names for Compound_ID, Protein_ID, and label_origin.
    """
    column_mappings = {
        'Compound_ID': ['SMILES', 'smiles', 'Smiles', 'compound', 'Compound', 'drug', 'smile', 'Drug'],
        'Protein_ID': ['Protein', 'protein', 'sequence', 'Sequence', 'target', 'Target'],
        'label_origin': ['Y', 'y', 'label', 'Label', 'value', 'Value']
    }

    reverse_map = {possible: standard
                   for standard, possibles in column_mappings.items()
                   for possible in possibles}

    rename_dict = {col: reverse_map[col] for col in df_ori.columns if col in reverse_map}

    if rename_dict:
        df_ori = df_ori.rename(columns=rename_dict)

    merge_keys = [key for key in ['Compound_ID', 'Protein_ID'] if key in df_ori.columns]

    # If neither key exists, create dummy IDs
    if not merge_keys:
        df_ori = df_ori.copy()
        df_ori.insert(0, 'Compound_ID', f"{file_prefix}_" + range(len(df_ori)).astype(str))
        df_ori.insert(1, 'Protein_ID', range(len(df_ori)))
        merge_keys = ['Compound_ID', 'Protein_ID']

    df_merged = pd.merge(concated_df, df_ori, on=merge_keys, how='left')
    return df_merged


def process_data(df, output_dir, csv_file, mode, models_mode, drug_id_file, prot_id_file):
    """
    Enrich the merged DataFrame with drug and protein IDs, then save the result.
    The output location depends on the mode value.
    """
    drug_df = pd.read_csv(drug_id_file)
    prot_df = pd.read_csv(prot_id_file)
    drug_df.columns = ['drugid', 'Compound_ID']
    prot_df.columns = ['protid', 'Protein_ID']

    df = df.merge(drug_df, on='Compound_ID', how='left')
    df = df.merge(prot_df, on='Protein_ID', how='left')

    if mode == 4:
        df.to_csv(f"{output_dir}/{csv_file}.csv", index=True, index_label='index')
    else:
        os.makedirs(f'{output_dir}/meta_data_{mode}_{models_mode[-1]}', exist_ok=True)
        df.to_csv(f"{output_dir}/meta_data_{mode}_{models_mode[-1]}/{csv_file}.csv",
                  index=True, index_label='index')


def main():
    data_dir = sys.argv[1]
    mode_1 = sys.argv[2]
    datatype = sys.argv[3]

    output_file = f'{data_dir}/data/end_merged'
    folder_path = f'{data_dir}/data/end_adaptivehit'
    csv_files = data_file_dir_csv(data_dir)

    os.makedirs(output_file, exist_ok=True)

    for csv_file in csv_files:
        df_ori = pd.read_csv(f'{data_dir}/{csv_file}.csv')

        # Currently only one mode is defined (4) with a list of model names.
        # The last element ('all') is used for file naming.
        modes = [(4, ['DeepConv-DTI', 'ConPLex', 'DrugBAN', 'all'])]

        for mode, models_mode in modes:
            concated_df = concat_csv_by_prefix(folder_path, csv_file, output_file, mode, models_mode)
            df_merged = label_csv_merge(concated_df, df_ori, csv_file, output_file)

            process_data(
                df_merged,
                output_file,
                csv_file,
                mode,
                models_mode,
                drug_id_file=f"{data_dir}/id/{datatype}_drugs.csv",
                prot_id_file=f"{data_dir}/id/{datatype}_prots.csv"
            )


if __name__ == "__main__":
    main()