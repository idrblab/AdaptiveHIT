# -*- coding: utf-8 -*-

import numpy as np
import os
from model import *
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import csv

num_atom_feat = 34
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom,explicit_H=False,use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                            Chem.rdchem.HybridizationType.SP2,
                            Chem.rdchem.HybridizationType.SP3,
                            Chem.rdchem.HybridizationType.SP3D,
                            Chem.rdchem.HybridizationType.SP3D2,
                            'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                one_of_k_encoding(atom.GetDegree(),degree) + \
                [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])


def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    #mol = Chem.AddHs(mol)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix

def txt_dir_path(folder_path):
    txt_file_paths = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            txt_file_paths.append(file_path)
    return txt_file_paths

# 预测文件格式读取
def csv_to_txt(csv_file, txt_file):
    df = pd.read_csv(csv_file)
    df.to_csv(txt_file, sep=' ', index=False, header=False, line_terminator='\n')

def featurizer_test(txt_file, dir_output):
    # csv_to_txt(csv_file, txt_file)
    with open(txt_file,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    failed_list = []
    # compounds_num = []
    # adj_num = []
    failed_count = 0
    compounds, adjacencies,proteins,interactions = [], [], [], []
    model = Word2Vec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "word2vec_30.model"))
    for no, data in enumerate(data_list):
        # print('/'.join(map(str, [no + 1, N])))
        smiles, sequence, interaction = data.strip().split(" ")
        try:
            protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
            proteins.append(protein_embedding)
            atom_feature, adj = mol_features(smiles)

            # print(atom_feature.shape)
            # compounds_num.append(atom_feature.shape)
            compounds.append(atom_feature)

            # print(adj.shape)
            # adj_num.append(adj.shape)
            adjacencies.append(adj)

            interactions.append(np.array([float(interaction)]))
        except KeyError:
            failed_count += 1
            failed_list.append(no)
            continue

    df_failed = pd.DataFrame(failed_list)
    df_failed.to_csv(fr'{dir_output}/failed.csv',index=False)

    # df_c = pd.DataFrame(compounds_num)
    # df_failed.to_csv(f'{dir_output}/compounds_num.csv')

    # df_a = pd.DataFrame(adj_num)
    # df_failed.to_csv(f'{dir_output}/adj_num.csv')

    print(dir_output, ' failed_count:', failed_count)
    # dir_input = ('dataset/' + DATASET + '/word2vec_30/')
    print(len(data_list), len(compounds),len(adjacencies),len(interactions),len(proteins))
    os.makedirs(dir_output, exist_ok=True)
    np.save(dir_output + '/compounds', compounds)
    np.save(dir_output + '/adjacencies', adjacencies)
    np.save(dir_output + '/proteins', proteins)
    np.save(dir_output + '/interactions', interactions)
    print('The preprocess of ' + dir_output + ' has finished!')


def featurizer_protein_test(csv_file_dir, dir_output):
    data = pd.read_csv(csv_file_dir)
    model = Word2Vec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "word2vec_30.model"))
    result = {}
    failed_protein_list = []
    failed_count = 0
    # 遍历每一行数据
    for index, row in data.iterrows():
        # 获取ID和数据
        id_value = row[0]
        sequence = row[1]
        print(id_value)
        try:
            protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
            result[id_value] = protein_embedding
        except KeyError:
            failed_count += 1
            failed_protein_list.append(id_value)
            continue
    np.save(dir_output + '/proteins', result)
    print('The preprocess of ' + dir_output + ' proteins has finished!')
    print('proteins num: ' + str(len(result)) + ' !')
    print(failed_protein_list)
    return failed_protein_list




def featurizer_compound_test(csv_file_dir, dir_output):
    data = pd.read_csv(csv_file_dir)
    result_compound = {}
    result_adj = {}
    failed_compound_adi_list = []
    failed_count = 0
    # 遍历每一行数据
    for index, row in data.iterrows():
        # 获取ID和数据
        id_value = row[0]
        compound = row[1]
        print(id_value)
        if '.' in compound:
            failed_count += 1
            failed_compound_adi_list.append(id_value)
            continue
        atom_feature, adj = mol_features(compound)
        result_compound[id_value] = atom_feature
        result_adj[id_value] = adj
    np.save(dir_output + '/compounds', result_compound)
    print('The preprocess of ' + dir_output + ' compounds has finished!')
    print('compounds num: ' + str(len(result_compound)) + ' !')
    np.save(dir_output + '/adjacencies', result_adj)
    print('The preprocess of ' + dir_output + ' adjacencies has finished!')
    print('adjacencies num: ' + str(len(result_adj)) + ' !')
    print(failed_compound_adi_list)
    return failed_compound_adi_list


def dti_genarate_test(csv_file_dir, failed_protein_list, failed_compound_adi_list, dir_output_train):
    # 读取原始 CSV 文件
    df_dti = pd.read_csv(csv_file_dir)
    filtered_df = df_dti[~df_dti['Compound_ID'].apply(lambda x: any(item in x for item in failed_compound_adi_list))]
    filtered_df = filtered_df[~filtered_df['Protein_ID'].apply(lambda x: any(item in x for item in failed_protein_list))]
    filtered_df.to_csv(f'{dir_output_train}/dti.csv',index=False)
    dti_num = len(filtered_df)
    return dti_num

def data_file_dir_txt(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    return csv_files

if __name__ == "__main__":
    # nohup python /public/home/lixy/..yanghao/646122/transformerCPI-master/GPCR/mol_featurizer_integ_retrain.py /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/DATA_bindingdb_similarity random > /public/home/lixy/..yanghao/646122/data_large/DATA_Retrain/DATA_bindingdb_similarity/TransformerCPI/myouts/mol_featurizer.file 2>&1 &   
    import argparse
    parser = argparse.ArgumentParser(description='none')
    # # # file_dir_params 
    parser.add_argument("data_dir", help="Training DTI information [drug, target, label]")
    parser.add_argument("mode", help="random or one")
    args = parser.parse_args()
    data_dir = args.data_dir
    mode = args.mode
    if mode == 'random':
        random_list = list(range(6,11))
        for random_name in random_list:
            mol_featurizer_list = ['train', 'dev', 'test']
            for mission_name in mol_featurizer_list:
                if mission_name == 'dev':
                    mission_name_deep = 'val'
                else:
                    mission_name_deep = mission_name

                dti_csv_file_dir = fr'{data_dir}/DeepConv-DTI/data/random_{random_name}/{mission_name_deep}/dti.csv'
                compound_csv_file_dir = fr'{data_dir}/DeepConv-DTI/data/random_{random_name}/{mission_name_deep}/compound.csv'
                protein_csv_file_dir = fr'{data_dir}/DeepConv-DTI/data/random_{random_name}/{mission_name_deep}/protein.csv'
                dir_output_train = fr'{data_dir}/TransformerCPI/data/random_{random_name}/{mission_name}'

                compounds_file = os.path.join(dir_output_train, 'compounds.npy')
                adjacencies_file = os.path.join(dir_output_train, 'adjacencies.npy')
                proteins_file = os.path.join(dir_output_train, 'proteins.npy')
                os.makedirs(dir_output_train, exist_ok=True)
                
                if os.path.exists(compounds_file) and os.path.exists(adjacencies_file) and os.path.exists(proteins_file):
                    print(f'Output files already exist in {dir_output_train}, skipping processing.')
                    continue 
                
                failed_compound_adi_list = featurizer_compound_test(compound_csv_file_dir, dir_output_train)
                print('failed_compound_adi_list is done!!!')
                failed_protein_list = featurizer_protein_test(protein_csv_file_dir, dir_output_train)
                dti_num = dti_genarate_test(dti_csv_file_dir, failed_protein_list, failed_compound_adi_list, dir_output_train)
                print('dti pair num:  ', str(dti_num), '!!!')

    else:
        mol_featurizer_list = ['train', 'dev', 'test']
        for mission_name in mol_featurizer_list:
            mission_name_deep = 'val' if mission_name == 'dev' else mission_name
            dti_csv_file_dir = fr'{data_dir}/DeepConv-DTI/data/{mission_name_deep}/dti.csv'
            compound_csv_file_dir = fr'{data_dir}/DeepConv-DTI/data/{mission_name_deep}/compound.csv'
            protein_csv_file_dir = fr'{data_dir}/DeepConv-DTI/data/{mission_name_deep}/protein.csv'
            dir_output_train = fr'{data_dir}/TransformerCPI/data/{mission_name}'

            os.makedirs(dir_output_train, exist_ok=True)
            failed_compound_adi_list = featurizer_compound_test(compound_csv_file_dir, dir_output_train)
            print('failed_compound_adi_list is done!!!')
            failed_protein_list = featurizer_protein_test(protein_csv_file_dir, dir_output_train)
            dti_num = dti_genarate_test(dti_csv_file_dir, failed_protein_list, failed_compound_adi_list, dir_output_train)
            print('dti pair num:  ', str(dti_num), '!!!')
