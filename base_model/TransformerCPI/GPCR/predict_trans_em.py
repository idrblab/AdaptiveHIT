# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import os
import time
from model import *
from mol_featurizer_new import *
import timeit
import pandas as pd
import json
import argparse


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

    # 修复 ragged array 警告
    compounds = np.array(compounds, dtype=object)
    adjacencies = np.array(adjacencies, dtype=object)
    proteins = np.array(proteins, dtype=object)
    interactions = np.array(interactions)
    return Compound_ID, Protein_ID, compounds, adjacencies, proteins, interactions


class Dti_prediction(object):
    def __init__(self, model):
        self.model = model
        # 添加一个标志来控制是否输出融合特征
        self.output_features = False
        self.feature_output_dir = None
    
    def set_output_features(self, value, output_dir=None):
        """设置是否输出融合特征和输出目录"""
        self.output_features = value
        self.feature_output_dir = output_dir
    
    def _extract_fused_features(self, compound, adj, protein, atom_num, protein_num):
        """
        从模型中提取融合特征（不修改原模型类）
        """
        # 获取模型中的各个组件
        encoder = self.model.encoder
        decoder = self.model.decoder
        weight = self.model.weight
        device = self.model.device
        
        # 获取最大长度
        compound_max_len = compound.shape[1]
        protein_max_len = protein.shape[1]
        
        # 创建mask
        compound_mask, protein_mask = self.model.make_masks(atom_num, protein_num, compound_max_len, protein_max_len)
        
        # GCN处理
        support = torch.matmul(compound, weight)
        compound_gcn = torch.bmm(adj, support)
        
        # 编码器处理
        enc_src = encoder(protein)
        
        # 解码器处理前的线性变换
        compound_ft = decoder.ft(compound_gcn)
        
        # 通过解码器层
        for layer in decoder.layers:
            compound_ft = layer(compound_ft, enc_src, compound_mask, protein_mask)
        
        # 计算注意力权重并进行加权求和（这就是融合特征）
        norm = torch.norm(compound_ft, dim=2)
        norm = F.softmax(norm, dim=1)
        
        fused_features = torch.zeros((compound_ft.shape[0], decoder.hid_dim)).to(device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = compound_ft[i, j, :]
                v = v * norm[i, j]
                fused_features[i, :] += v
        
        return fused_features
    
    def predict(self, dataset):
        self.model.eval()
        N = len(dataset)
        result_df = {}
        D, P, T, Y, S = [], [], [], [], []
        # 添加存储融合特征的列表
        F = []
        
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins, labels = [], [], [], []
                Compound_ID, Protein_ID, atom, adj, protein, label = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                
                # 使用原有的pack函数打包数据
                packed_data = pack(atoms, adjs, proteins, labels, self.model.device)
                compound, adj, protein, correct_interaction, atom_num, protein_num = packed_data
                
                if self.output_features:
                    # 获取预测结果（使用原模型）
                    correct_labels, predicted_labels, predicted_scores = self.model(packed_data, train=False)
                    
                    # 单独提取融合特征（不修改原模型）
                    fused_features = self._extract_fused_features(compound, adj, protein, atom_num, protein_num)
                    F.append(fused_features.cpu().numpy())
                else:
                    correct_labels, predicted_labels, predicted_scores = self.model(packed_data, train=False)
                
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
                D.append(Compound_ID)
                P.append(Protein_ID)
        
        result_df['Compound_ID'] = D
        result_df['Protein_ID'] = P
        result_df['Predicted_scores'] = S
        result_df['label_predict'] = [1 if x >= 0.5 else 0 for x in S]
        
        # 如果启用了特征输出，保存特征到独立文件
        if self.output_features and len(F) > 0 and self.feature_output_dir:
            self._save_features(F, result_df)
        
        return result_df
    
    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        result_df = {}
        D, P, T, Y, S = [], [], [], [], []
        # 添加存储融合特征的列表
        F = []
        
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins, labels = [], [], [], []
                Compound_ID, Protein_ID, atom, adj, protein, label = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                
                # 使用原有的pack函数打包数据
                packed_data = pack(atoms, adjs, proteins, labels, self.model.device)
                compound, adj, protein, correct_interaction, atom_num, protein_num = packed_data
                
                if self.output_features:
                    # 获取预测结果（使用原模型）
                    correct_labels, predicted_labels, predicted_scores = self.model(packed_data, train=False)
                    
                    # 单独提取融合特征（不修改原模型）
                    fused_features = self._extract_fused_features(compound, adj, protein, atom_num, protein_num)
                    F.append(fused_features.cpu().numpy())
                else:
                    correct_labels, predicted_labels, predicted_scores = self.model(packed_data, train=False)
                
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
        Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        F1 = 2 * Pre * Rec / (Pre + Rec) if (Pre + Rec) > 0 else 0
        MCC = matthews_corrcoef(T, Y)
        precision, recall, _ = precision_recall_curve(T, S)
        PRC = auc(recall, precision)
        
        # 如果启用了特征输出，保存特征到独立文件
        if self.output_features and len(F) > 0 and self.feature_output_dir:
            self._save_features(F, result_df)
        
        return result_df, AUC, PRC, ACC, Rec, Pre, F1, MCC
    
    def _save_features(self, features_list, result_df):
        """
        保存特征到独立文件
        """
        # 将特征列表转换为numpy数组
        feature_array = np.array(features_list)
        
        # 使用test_name作为文件名
        base_name = os.path.join(self.feature_output_dir, f"features_{args.test_name}")
        
        # 保存特征数组
        feature_file = f"{base_name}.npy"
        np.save(feature_file, feature_array)
        
        # 创建索引映射，包含特征文件名信息
        result_df['feature_index'] = range(len(features_list))
        result_df['feature_file'] = feature_file
        
        # # 保存索引映射文件
        # index_file = f"{base_name}_index.csv"
        # index_df = pd.DataFrame({
        #     'Compound_ID': result_df['Compound_ID'],
        #     'Protein_ID': result_df['Protein_ID'],
        #     'feature_index': range(len(features_list)),
        #     'feature_file': os.path.basename(feature_file)  # 添加特征文件名列
        # })
        # index_df.to_csv(index_file, index=False)
        
        # # 保存元数据
        # meta_file = f"{base_name}_meta.json"
        # meta_data = {
        #     'test_name': self.test_name,
        #     'feature_file': os.path.basename(feature_file),
        #     'index_file': os.path.basename(index_file),
        #     'num_samples': len(features_list),
        #     'feature_dim': feature_array.shape[1] if len(feature_array.shape) > 1 else 1,
        #     'dtype': str(feature_array.dtype)
        # }
        # with open(meta_file, 'w') as f:
        #     json.dump(meta_data, f, indent=2)
        
        print(f"特征已保存到: {feature_file}")
        # print(f"索引文件: {index_file}")
        # print(f"元数据: {meta_file}")

def load_tensor(array, dtype):
    return [dtype(d).to(device) for d in array]

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def list_cuda_to_cpu(list_cuda):
    cuda_device = torch.device('cuda:0')
    list_cpu = [tensor.tolist() for tensor in list_cuda]
    list_cpu = [torch.tensor(names).cpu() for names in list_cpu]
    return list_cpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This Python script is used to train, validate, test deep learning model for prediction of drug-target interaction                                 
    """)
    # datasets_params
    parser.add_argument("model_dir", help="dir of model ")
    parser.add_argument("mode", help="mode of model")
    parser.add_argument("test_name", help="name of test ")
    parser.add_argument("dataset_test_dir", help="dir of dataset_test [SMILES, sqeuence, label]")
    
    # 新增参数：控制是否输出融合特征
    parser.add_argument("--output_features", action="store_true", 
                        help="Output fused features before classifier as binary files")
    parser.add_argument("--output", "-o", help="Prediction output directory", type=str, default="./output/")
    
    args = parser.parse_args()

    # 设置随机种子
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)

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
    print("dataset_test load success")

    """ load model """
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
    
    # 确保输出目录存在
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    Predict = Dti_prediction(model)
    
    # 根据命令行参数设置是否输出融合特征
    if args.output_features:
        Predict.set_output_features(True, output_dir)
        print("融合特征输出模式已开启，特征将保存为二进制文件")
    else:
        Predict.set_output_features(False)
        print("融合特征输出模式已关闭")

    if args.mode == 'predict':
        result_df = Predict.predict(dataset_test)
        
        # 构建输出字典
        predict_dict = {
            'Compound_ID': result_df['Compound_ID'], 
            'Protein_ID': result_df['Protein_ID'], 
            'Predicted_scores': result_df['Predicted_scores'], 
            'label_predict': result_df['label_predict']
        }
        
        # 如果有特征索引，添加到输出
        if args.output_features and 'feature_index' in result_df:
            predict_dict['feature_index'] = result_df['feature_index']
            predict_dict['feature_file'] = f'features_{args.test_name}'
        
        predict_df = pd.DataFrame(predict_dict)
        predict_df.to_csv(os.path.join(output_dir, f'{args.test_name}_score.csv'), index=False)
    else:
        result_df, AUC, PRC, ACC, Rec, Pre, F1, MCC = Predict.test(dataset_test)
        
        # 构建输出字典
        predict_dict = {
            'Compound_ID': result_df['Compound_ID'], 
            'Protein_ID': result_df['Protein_ID'], 
            'Predicted_scores': result_df['Predicted_scores'], 
            'label_predict': result_df['label_predict'], 
            'label_original': result_df['label_original']
        }
        
        # 如果有特征索引，添加到输出
        if args.output_features and 'feature_index' in result_df:
            predict_dict['feature_index'] = result_df['feature_index']
        
        predict_df = pd.DataFrame(predict_dict)
        predict_df.to_csv(os.path.join(output_dir, f'{args.test_name}_score.csv'), index=False)
        
        result_dic = {"test_name":[args.test_name], "AUC":[AUC], "AUPR": [PRC], "ACC":[ACC], 
                     "Rec":[Rec], "Pre":[Pre], "F1":[F1], "MCC":[MCC]}
        df_eva = pd.DataFrame(result_dic)
        df_eva.to_csv(os.path.join(output_dir, 'evaluation.csv'), index=False, mode='a')