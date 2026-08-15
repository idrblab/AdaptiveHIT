# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import os
import time
from model import *
from mol_featurizer_integ_retrain import *
import timeit
import pandas as pd




# 预测文件格式读取
def csv_to_txt(csv_file, txt_file):
    df = pd.read_csv(csv_file)
    df.to_csv(txt_file, sep=' ', index=False, header=False, line_terminator='\n')


def featurizer_predict(txt_file):
    # csv_to_txt(csv_file, txt_file)
    with open(txt_file,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)

    Compound_ID, Protein_ID, compounds, adjacencies,proteins,interactions = [], [], [], [], [], []
    model = Word2Vec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "word2vec_30.model"))
    for no, data in enumerate(data_list):
        # print('/'.join(map(str, [no + 1, N])))
        try:
            smiles, sequence, interaction = data.strip().split(" ")
        except Exception as e:
            print(f"Line {no} : {data}")
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to parse SMILES: {smiles}")
            continue
        Compound_ID.append(smiles)
        Protein_ID.append(sequence)
        atom_feature, adj = mol_features(smiles)
        compounds.append(atom_feature)
        adjacencies.append(adj)
        interactions.append(np.array([float(interaction)]))
        try:
            protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
        except Exception as e:
            print(f"处理序列时出现异常: {e}")
            print(f"出现问题的序列是: {sequence}")
            continue
        proteins.append(protein_embedding)

    compounds = np.array(compounds)
    adjacencies = np.array(adjacencies)
    proteins = np.array(proteins)
    interactions = np.array(interactions)
    # # dir_input = ('dataset/' + DATASET + '/word2vec_30/')
    # # print(len(data_list), len(compounds),len(adjacencies),len(interactions),len(proteins))
    # os.makedirs(dir_input, exist_ok=True)
    # np.save(dir_input + 'compounds', compounds)
    # np.save(dir_input + 'adjacencies', adjacencies)
    # np.save(dir_input + 'proteins', proteins)
    # np.save(dir_input + 'interactions', interactions)
    return Compound_ID, Protein_ID, compounds, adjacencies, proteins, interactions


class Dti_prediction(object):
    def __init__(self, model):
        self.model = model
    
    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        result_df = {}
        D, P, T, Y, S = [], [], [], [], []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins, labels = [], [], [], []
                Compound_ID, Protein_ID, atom, adj, protein, label = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                data = pack(atoms,adjs,proteins, labels, self.model.device)
                correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
                D.append(Compound_ID)
                P.append(Protein_ID)

        result_df['Compound_ID'] = D
        result_df['Protein_ID'] = P
        result_df['Predicted_scores'] = S
        result_df['label_predict'] = [1 if x >= 0.5 else 0 for x in S]
        result_df['label_original'] = T

        ACC = accuracy_score(T, Y)
        AUC = roc_auc_score(T, S)
        CM = confusion_matrix(T, Y)
        TN = CM[0][0]
        FP = CM[0][1]
        FN = CM[1][0]
        TP = CM[1][1]
        Rec = TP / (TP + FN)
        Pre = TP / (TP + FP)
        F1 = 2 * Pre * Rec / (Pre + Rec)
        MCC = matthews_corrcoef(T, Y)
        precision, recall, _ = precision_recall_curve(T, S)
        PRC = auc(recall, precision)
        return result_df, AUC, PRC, ACC, Rec, Pre, F1, MCC

    

    def predict(self, dataset):
        self.model.eval()
        N = len(dataset)
        result_df = {}
        D, P, T, Y, S = [], [], [], [], []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins, labels = [], [], [], []
                Compound_ID, Protein_ID, atom, adj, protein, label = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                data = pack(atoms,adjs,proteins, labels, self.model.device)
                correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
                D.append(Compound_ID)
                P.append(Protein_ID)
        result_df['Compound_ID'] = D
        result_df['Protein_ID'] = P
        result_df['Predicted_scores'] = S
        result_df['label_predict'] = [1 if x >= 0.5 else 0 for x in S]
        return result_df

                # result_columns = ['Compound_ID', 'Protein_ID','predicted', "label_predict"]
                # temp_df[dataset, 'predicted'] = predicted
                # temp_df[dataset, 'Compound_ID'] = prediction_dic["Compound_ID"]
                # temp_df[dataset, 'Protein_ID'] = prediction_dic["Protein_ID"]
                # result_columns.append((data, "label_predict"))

                # if with_label:
                #     result_columns.append((dataset, "label_original"))

def load_tensor(array, dtype):
    return [dtype(d).to(device) for d in array]

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def list_cuda_to_cpu(list_cuda):
    cuda_device = torch.device('cuda:0')
    # tensor_cuda = [torch.from_numpy(arr).cuda() for arr in list_cuda]
    # list_cuda = [tensor.numpy() for tensor in list_cuda]
    list_cpu = [tensor.tolist() for tensor in list_cuda]
    list_cpu = [torch.tensor(names).cpu() for names in list_cpu]
    # list_cpu = [tensor.cpu().tolist() for tensor in list_cpu]
    return list_cpu



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""
    This Python script is used to train, validate, test deep learning model for prediction of drug-target interaction                                 
    """)
    # datasets_params
    parser.add_argument("model_dir", help="dir of model ")
    parser.add_argument("mode", help="mode of model")
    parser.add_argument("test_name", help="name of test ")
    parser.add_argument("dataset_test_dir", help="dir of dataset_test [SMILES, sqeuence, label]")

    # output_params
    parser.add_argument("--output", "-o", help="Prediction output", type=str)
    args = parser.parse_args()

# nuhup python predict.py model_dir test bindingbd/test dataset_test_dir -o ./output/$data_dir/output
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    DATASET = "GPCR_train"

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """preproce data and Load preprocessed test_data."""
    dir_input_test = args.dataset_test_dir
    Compound_ID, Protein_ID, compounds, adjacencies, proteins, interactions = featurizer_predict(dir_input_test)

    compounds = load_tensor(compounds, torch.FloatTensor)
    proteins = load_tensor(proteins, torch.FloatTensor)
    interactions = load_tensor(interactions, torch.FloatTensor)
    adjacencies = load_tensor(adjacencies, torch.FloatTensor)

    dataset_test = list(zip(Compound_ID, Protein_ID, compounds, adjacencies, proteins, interactions))
    # dataset_test = shuffle_dataset(dataset_test, 0)
    print("dataset_test load success")

    """ loda model """
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    kernel_size = 7
    file_model = args.model_dir

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.load_state_dict(torch.load(file_model, map_location=device))
    model.to(device)
    
    print("model load success")
    
    """Output"""
    model.eval()
    output_file = args.output
    Predict = Dti_prediction(model)

    if args.mode == 'predict':
        result_df = Predict.predict(dataset_test)


        predict_dict = {'Compound_ID': result_df['Compound_ID'], 'Protein_ID': result_df['Protein_ID'], 'Predicted_scores': result_df['Predicted_scores'], 'label_predict':result_df['label_predict']}
        # print('Compound_ID:', len(result_df['Compound_ID']))
        # print('Compound_ID:', result_df['Compound_ID'])
        # print('Protein_ID:', len(result_df['Protein_ID']))
        # print('Protein_ID:', result_df['Protein_ID'])
        # print('Predicted_scores:', len(result_df['Predicted_scores']))
        # print('Predicted_scores:', result_df['Predicted_scores'])
        # print('label_predict:', len(result_df['label_predict']))
        # print('label_predict:', result_df['label_predict'])
        predict_df = pd.DataFrame(predict_dict)
        predict_df.to_csv(f'{output_file}{args.test_name}_score.csv', index=False)
    else:
        result_df, AUC, PRC, ACC, Rec, Pre, F1, MCC = Predict.test(dataset_test)
        # print(type(result_df["Compound_ID"]))
        # Compound_ID = list_cuda_to_cpu(result_df["Compound_ID"])
        # print(Compound_ID)
        # print(type(Compound_ID))
        # Protein_ID = list_cuda_to_cpu(result_df["Protein_ID"])
        # Predicted_scores = list_cuda_to_cpu(result_df["Predicted_scores"])
        # label_predict = list_cuda_to_cpu(result_df["label_predict"])
        # label_original = list_cuda_to_cpu(result_df["label_original"])

        predict_dict = {'Compound_ID': result_df['Compound_ID'], 'Protein_ID': result_df['Protein_ID'], 'Predicted_scores': result_df['Predicted_scores'], 'label_predict':result_df['label_predict'], 'label_original':result_df['label_original']}
        # predict_dict = {'Predicted_scores': result_df['Predicted_scores'], 'label_predict':result_df['label_predict'], 'label_original':result_df['label_original']}

        predict_df = pd.DataFrame(predict_dict)
        predict_df.to_csv(f'{output_file}{args.test_name}_score.csv', index=False)

        result_dic = {"test_name":[args.test_name], "AUC":[AUC], "AUPR": [PRC], "ACC":[ACC], "Rec":[Rec], "Pre":[Pre], "F1":[F1], "MCC":[MCC]}
        df_eva = pd.DataFrame(result_dic)
        df_eva.to_csv(f'{output_file}evaluation.csv', index=False, mode='a')

