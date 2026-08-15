import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model, Model
import argparse
import h5py
from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
import time
import json
import os


seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}
def evaluation(test_label, predicted_labels, prediction):
    ACC = accuracy_score(test_label, predicted_labels)
    AUC = roc_auc_score(test_label, prediction)
    CM = confusion_matrix(test_label, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    F1 = 2 * Pre * Rec / (Pre + Rec) if (Pre + Rec) > 0 else 0
    MCC = matthews_corrcoef(test_label, predicted_labels)
    precision, recall, _ = precision_recall_curve(test_label, prediction)
    PRC = auc(recall, precision)
    return ACC, AUC, Rec, Pre, F1, MCC, PRC

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0] 
    else:
        return [seq_dic[aa] for aa in seq]

def parse_data(dti_dir, drug_dir, protein_dir, with_label=True,
               prot_len=2500, prot_vec="Convolution",
               drug_vec="Convolution", drug_len=2048):

    print("Parsing {0} , {1}, {2} with length {3}, type {4}".format(dti_dir, drug_dir, protein_dir, prot_len, prot_vec))

    protein_col = "Protein_ID"
    drug_col = "Compound_ID"
    col_names = [protein_col, drug_col]
    if with_label:
        label_col = "Label"
        col_names += [label_col]
    dti_df = pd.read_csv(dti_dir)
    drug_df = pd.read_csv(drug_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")


    if prot_vec == "Convolution":
        protein_df["encoded_sequence"] = protein_df.Sequence.map(lambda a: encodeSeq(a, seq_dic))
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)
    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True)
    drug_feature = np.stack(dti_df[drug_vec].map(lambda fp: fp.split("\t")))
    if prot_vec=="Convolution":
        protein_feature = sequence.pad_sequences(dti_df["encoded_sequence"].values, prot_len)
    else:
        protein_feature = np.stack(dti_df[prot_vec].map(lambda fp: fp.split("\t")))
    if with_label:
        label = dti_df[label_col].values
        print("\tPositive data : %d" %(sum(dti_df[label_col])))
        print("\tNegative data : %d" %(dti_df.shape[0] - sum(dti_df[label_col])))
        return {"protein_feature": protein_feature.astype(np.float32), 
                "drug_feature": drug_feature.astype(np.float32), 
                "label": label,
                "Compound_ID": dti_df["SMILES"].tolist(), 
                "Protein_ID": dti_df["Sequence"].tolist()}
    else:
        return {"protein_feature": protein_feature.astype(np.float32), 
                "drug_feature": drug_feature.astype(np.float32),
                "Compound_ID": dti_df["SMILES"].tolist(), 
                "Protein_ID": dti_df["Sequence"].tolist()}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--mode", '-m', help="mode of model", default='test', type=str)
    parser.add_argument("--with-label", "-W", help="Existence of label information in test DTI", default='N', type=str)
    # test_params
    parser.add_argument("--test-name", '-n', help="Name of test data sets", type=str)
    parser.add_argument("--test-dti-dir", "-i", help="Test dti [drug, target, [label]]", type=str)
    parser.add_argument("--test-drug-dir", "-d", help="Test drug information [drug, SMILES,[feature_name, ..]]", type=str)
    parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]", type=str)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)
    
    # 新增参数：是否输出融合特征（与之前模型保持一致）
    parser.add_argument("--output_features", action="store_true", 
                        help="Output fused features before classifier as binary files")

    parser.add_argument("--prot-vec", "-v", help="Type of protein feature, if Convolution, it will execute conlvolution on sequeunce", type=str, default="Convolution")
    parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)
    parser.add_argument("--drug-vec", "-V", help="Type of drug feature", type=str, default="morgan_fp")
    parser.add_argument("--drug-len", "-L", help="Drug vector length", default=2048, type=int)
    args = parser.parse_args()
    
    model = args.model
    test_name = args.test_name
    test_dti = args.test_dti_dir
    test_protein = args.test_protein_dir
    test_drug = args.test_drug_dir
    
    mode = args.mode
    output_file = args.output
    with_label = args.with_label
    if with_label == 'T':
        with_label = True
    else:
        with_label = False

    # 删除优化器权重以减小模型大小
    try:
        f = h5py.File(model, 'r+')
        try:
            f.__delitem__("optimizer_weights")
        except:
            print("optimizer_weights are already deleted")
        f.close()
    except Exception as e:
        print(f"Warning: Could not modify h5 file: {e}")

    type_params = {
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
        "drug_vec": args.drug_vec,
        "drug_len": args.drug_len,
    }
    
    # 解析测试数据
    test_dic = {test_name: parse_data(test_dti, test_drug, test_protein, with_label=with_label, **type_params)}
    
    # 加载模型
    if tf.test.is_gpu_available():
        with tf.device('/gpu:0'):
            loaded_model = load_model(model)
    else:
        with tf.device('/cpu:0'):
            loaded_model = load_model(model)
    
    # 如果需要输出特征，创建特征提取模型
    feature_extractor = None
    if args.output_features:
        # 找到融合层（Concatenate层）的输出
        # 根据模型结构，融合层是Concatenate层，它将药物和蛋白质特征连接起来
        for i, layer in enumerate(loaded_model.layers):
            if isinstance(layer, keras.layers.Concatenate):
                print(f"Found Concatenate layer at index {i}, name: {layer.name}")
                # 创建特征提取模型，输出Concatenate层的输出
                feature_extractor = Model(inputs=loaded_model.input,
                                          outputs=layer.output)
                break
        
        if feature_extractor is None:
            print("Warning: Could not find Concatenate layer, will try to use the layer before output")
            # 如果找不到Concatenate层，使用倒数第二层
            feature_extractor = Model(inputs=loaded_model.input,
                                      outputs=loaded_model.layers[-2].output)
        
        print("Feature extraction model created - will save fusion features as binary files")

    print("prediction")
    for dataset in test_dic:
        prediction_dic = test_dic[dataset]
        
        # 确保数据类型正确
        drug_feature = prediction_dic["drug_feature"].astype(np.float32)
        protein_feature = prediction_dic["protein_feature"].astype(np.float32)
        
        N = int(np.ceil(drug_feature.shape[0] / 50))
        d_splitted = np.array_split(drug_feature, N)
        p_splitted = np.array_split(protein_feature, N)
        
        # 存储所有批次的预测结果和特征
        all_predictions = []
        all_features = []
        
        for d, p in zip(d_splitted, p_splitted):
            # 预测
            pred = loaded_model.predict([d, p])
            pred = np.squeeze(pred)
            if pred.ndim == 0:
                pred = [float(pred)]
            else:
                pred = pred.tolist()
            all_predictions.extend(pred)
            
            # 如果需要特征，提取融合特征
            if args.output_features and feature_extractor is not None:
                features = feature_extractor.predict([d, p])
                all_features.append(features)
        
        # 如果需要特征，合并所有特征
        feature_file_name = None
        if args.output_features and all_features and feature_extractor is not None:
            # 合并所有批次的特征
            all_features = np.vstack(all_features)
            
            # 确保特征数量与预测结果数量一致
            assert len(all_features) == len(all_predictions), \
                f"Feature count ({len(all_features)}) doesn't match prediction count ({len(all_predictions)})"
            
            # 创建输出目录（如果不存在）
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存特征为npy文件（二进制格式）
            feature_file_name = f"features_{test_name}.npy"
            feature_file_path = os.path.join(os.path.dirname(output_file), feature_file_name)
            np.save(feature_file_path, all_features)
            
            # # 保存特征元数据
            # meta_file_path = os.path.join(os.path.dirname(output_file), f"features_{test_name}_meta.json")
            # meta_data = {
            #     'test_name': test_name,
            #     'feature_file': feature_file_name,
            #     'num_samples': len(all_features),
            #     'feature_dim': all_features.shape[1] if len(all_features.shape) > 1 else 1,
            #     'dtype': str(all_features.dtype),
            #     'model': args.model,
            #     'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            # }
            # with open(meta_file_path, 'w') as f:
            #     json.dump(meta_data, f, indent=2)
            
            print(f"Fusion features saved to: {feature_file_path}")
            # print(f"Feature metadata saved to: {meta_file_path}")
        
        if mode == 'predict':
            predict_dict = {
                'Compound_ID': prediction_dic["Compound_ID"], 
                'Protein_ID': prediction_dic["Protein_ID"], 
                'Predicted_scores': all_predictions, 
                'label_predict': [1 if x >= 0.5 else 0 for x in all_predictions]
            }
            
            # 如果启用了特征输出，添加特征索引和文件名
            if args.output_features and feature_file_name is not None:
                predict_dict['feature_index'] = list(range(len(all_predictions)))
                predict_dict['feature_file'] = feature_file_name
            
            output_path = f'{output_file}{test_name}_score.csv'
            print("save to %s" % output_path)
            result_df = pd.DataFrame(predict_dict)
            result_df.to_csv(output_path, index=False)
            
        else:
            label_original = np.squeeze(test_dic[dataset]['label'])
            predict_dict = {
                'Compound_ID': prediction_dic["Compound_ID"], 
                'Protein_ID': prediction_dic["Protein_ID"], 
                'Predicted_scores': all_predictions, 
                'label_predict': [1 if x >= 0.5 else 0 for x in all_predictions], 
                'label_original': label_original.tolist() if hasattr(label_original, 'tolist') else label_original
            }
            
            # 如果启用了特征输出，添加特征索引和文件名
            if args.output_features and feature_file_name is not None:
                predict_dict['feature_index'] = list(range(len(all_predictions)))
                predict_dict['feature_file'] = feature_file_name
            
            output_path = f'{output_file}{test_name}_score.csv'
            print("save to %s" % output_path)
            result_df = pd.DataFrame(predict_dict)
            result_df.to_csv(output_path, index=False)
            
            ACC, AUC, Rec, Pre, F1, MCC, PRC = evaluation(
                predict_dict['label_original'], 
                predict_dict['label_predict'], 
                predict_dict['Predicted_scores']
            )
            result_dic = {
                "test_name": [test_name], 
                "AUC": [AUC], 
                "AUPR": [PRC], 
                "ACC": [ACC], 
                "Rec": [Rec], 
                "Pre": [Pre], 
                "F1": [F1], 
                "MCC": [MCC]
            }
            df_eva = pd.DataFrame(result_dic)
            df_eva.to_csv(f'{output_file}evaluation.csv', index=False, mode='a')