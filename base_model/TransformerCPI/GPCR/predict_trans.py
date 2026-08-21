# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import os
import time
import timeit
import pandas as pd
from model import *
from mol_featurizer_integ_retrain import *



def csv_to_txt(csv_file, txt_file):
    """Convert a CSV file to space-separated TXT (no header)."""
    df = pd.read_csv(csv_file)
    df.to_csv(txt_file, sep=' ', index=False, header=False, line_terminator='\n')


def featurizer_predict(txt_file):
    """
    Read molecule-protein pairs from a text file,
    featurize them using RDKit and Word2Vec protein embeddings.
    """
    with open(txt_file, "r") as f:
        data_list = f.read().strip().split('\n')
    # Exclude entries containing '.' in the SMILES (likely salts)
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]

    Compound_ID, Protein_ID, compounds, adjacencies, proteins, interactions = [], [], [], [], [], []
    # Load pre-trained Word2Vec model for protein sequences
    word2vec_model_path = os.path.join(os.path.dirname(__file__), "word2vec_30.model")
    model = Word2Vec.load(word2vec_model_path)

    for no, data in enumerate(data_list):
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
            print(f"Exception processing sequence: {e}")
            print(f"Problematic sequence: {sequence}")
            continue
        proteins.append(protein_embedding)

    compounds = np.array(compounds)
    adjacencies = np.array(adjacencies)
    proteins = np.array(proteins)
    interactions = np.array(interactions)

    return Compound_ID, Protein_ID, compounds, adjacencies, proteins, interactions


class Dti_prediction(object):
    """Prediction class for DTI models."""
    
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        """Evaluate the model on a test set with known labels."""
        self.model.eval()
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

                data = pack(atoms, adjs, proteins, labels, self.model.device)
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
        """Generate predictions for unlabeled data."""
        self.model.eval()
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

                data = pack(atoms, adjs, proteins, labels, self.model.device)
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


def load_tensor(array, dtype):
    """Convert a list of numpy arrays to a list of torch tensors on the device."""
    return [dtype(d).to(device) for d in array]


def shuffle_dataset(dataset, seed):
    """Shuffle a dataset with a fixed seed."""
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def list_cuda_to_cpu(list_cuda):
    """Convert a list of CUDA tensors to CPU tensors."""
    list_cpu = [tensor.tolist() for tensor in list_cuda]
    list_cpu = [torch.tensor(names).cpu() for names in list_cpu]
    return list_cpu


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train, validate, or test a DTI prediction model.")
    
    parser.add_argument("model_dir", help="Directory containing the trained model file")
    parser.add_argument("mode", help="Mode: 'predict' or 'test'")
    parser.add_argument("test_name", help="Name identifier for the test set")
    parser.add_argument("dataset_test_dir", help="Path to the test dataset file (SMILES, sequence, label)")
    parser.add_argument("--output", "-o", help="Output directory for prediction results", type=str)

    args = parser.parse_args()

    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    # Load and featurize the test data
    Compound_ID, Protein_ID, compounds, adjacencies, proteins, interactions = featurizer_predict(args.dataset_test_dir)

    compounds = load_tensor(compounds, torch.FloatTensor)
    proteins = load_tensor(proteins, torch.FloatTensor)
    interactions = load_tensor(interactions, torch.FloatTensor)
    adjacencies = load_tensor(adjacencies, torch.FloatTensor)

    dataset_test = list(zip(Compound_ID, Protein_ID, compounds, adjacencies, proteins, interactions))
    print("dataset_test loaded successfully.")

    # Load the trained model
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    kernel_size = 7

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.load_state_dict(torch.load(args.model_dir, map_location=device))
    model.to(device)

    print("Model loaded successfully.")

    # Perform prediction or testing
    predictor = Dti_prediction(model)

    if args.mode == 'predict':
        result_df = predictor.predict(dataset_test)
        predict_dict = {
            'Compound_ID': result_df['Compound_ID'],
            'Protein_ID': result_df['Protein_ID'],
            'Predicted_scores': result_df['Predicted_scores'],
            'label_predict': result_df['label_predict']
        }
        predict_df = pd.DataFrame(predict_dict)
        predict_df.to_csv(f'{args.output}{args.test_name}_score.csv', index=False)
    else:
        result_df, AUC, PRC, ACC, Rec, Pre, F1, MCC = predictor.test(dataset_test)
        predict_dict = {
            'Compound_ID': result_df['Compound_ID'],
            'Protein_ID': result_df['Protein_ID'],
            'Predicted_scores': result_df['Predicted_scores'],
            'label_predict': result_df['label_predict'],
            'label_original': result_df['label_original']
        }
        predict_df = pd.DataFrame(predict_dict)
        predict_df.to_csv(f'{args.output}{args.test_name}_score.csv', index=False)

        result_dic = {
            "test_name": [args.test_name],
            "AUC": [AUC],
            "AUPR": [PRC],
            "ACC": [ACC],
            "Rec": [Rec],
            "Pre": [Pre],
            "F1": [F1],
            "MCC": [MCC]
        }
        df_eva = pd.DataFrame(result_dic)
        df_eva.to_csv(f'{args.output}evaluation.csv', index=False, mode='a')