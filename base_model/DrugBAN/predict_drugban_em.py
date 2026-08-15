comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from models import DrugBAN
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Predicter
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd
import numpy as np
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")

# training_params batch_size 
parser.add_argument("batch_size", help="batch_size of model", type=int)
parser.add_argument("model_dir", help="dir of model ")
parser.add_argument("mode", help="mode of model")
parser.add_argument("test_name", help="name of test ")
parser.add_argument("dataset_test_dir", help="dir of dataset_test [SMILES, sqeuence, label]")

# output_params
parser.add_argument("--output", "-o", help="Prediction output", type=str)
# 新增参数：是否输出融合特征（与上一个模型保持一致）
parser.add_argument("--output_features", action="store_true", 
                    help="Output fused features before classifier as binary files")

args = parser.parse_args()
batch_size = args.batch_size
dataset_dir = args.dataset_test_dir

# 定义hook函数来捕获融合特征
fusion_features_list = []

def hook_fn(module, input, output):
    """
    Hook function to capture fusion features (f) from BAN layer
    output: tuple (f, att) - f is the fusion features, att is attention weights
    """
    # output是一个元组 (f, att)，我们需要第一个元素f
    f = output[0]  # 提取融合特征
    fusion_features_list.append(f.detach().cpu().numpy())

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.SOLVER.BATCH_SIZE = batch_size
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]

    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    # 读取测试数据
    if not cfg.DA.TASK:
        df_test = pd.read_csv(dataset_dir)
        test_dataset = DTIDataset(df_test.index.values, df_test)
    else:
        test_target_path = os.path.join(dataFolder, 'target_test.csv')
        df_test_target = pd.read_csv(test_target_path)
        test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    if not cfg.DA.USE:
        params['shuffle'] = False
        params['drop_last'] = False
        if not cfg.DA.TASK:
            test_generator = DataLoader(test_dataset, **params)
        else:
            test_generator = DataLoader(test_target_dataset, **params)
    else:
        params['shuffle'] = False
        params['drop_last'] = False
        test_generator = DataLoader(test_target_dataset, **params)
        
    model = DrugBAN(**cfg).to(device)

    torch.backends.cudnn.benchmark = True
    state_dict = torch.load(args.model_dir, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print('model load success')

    # 如果需要保存特征，注册hook到bcn模块
    handle = None
    if args.output_features:
        # 注册hook到bcn模块（输出融合特征f）
        handle = model.bcn.register_forward_hook(hook_fn)
        print("Feature extraction hook registered - will save fusion features as binary files")

    if not cfg.DA.USE:
        Tester = Predicter(model, device, test_generator, **cfg)
    else:
        Tester = Predicter(model, device, test_generator, **cfg)
    
    if args.mode == 'predict':    
        test_loss, correct_labels, predicted_labels, predicted_scores = Tester.predict()
        print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")
        
        # 如果保存了特征，处理并保存（与上一个模型保持一致）
        feature_file_name = None
        if args.output_features and fusion_features_list and handle is not None:
            # 移除hook
            handle.remove()
            
            # 合并所有batch的特征
            all_features = np.vstack(fusion_features_list)
            
            # 确保特征数量与预测结果数量一致
            assert len(all_features) == len(predicted_scores), \
                f"Feature count ({len(all_features)}) doesn't match prediction count ({len(predicted_scores)})"
            
            # 创建输出目录（如果不存在）
            os.makedirs(args.output, exist_ok=True)
            
            # 保存特征为npy文件（二进制格式）
            feature_file_name = f"features_{args.test_name}.npy"
            feature_file_path = os.path.join(args.output, feature_file_name)
            np.save(feature_file_path, all_features)
            
            # # 保存特征元数据
            # meta_file_path = os.path.join(args.output, f"features_{args.test_name}_meta.json")
            # meta_data = {
            #     'test_name': args.test_name,
            #     'feature_file': feature_file_name,
            #     'num_samples': len(all_features),
            #     'feature_dim': all_features.shape[1] if len(all_features.shape) > 1 else 1,
            #     'dtype': str(all_features.dtype),
            #     'model_dir': args.model_dir,
            #     'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            # }
            # with open(meta_file_path, 'w') as f:
            #     json.dump(meta_data, f, indent=2)
            
            print(f"Fusion features saved to: {feature_file_path}")
            # print(f"Feature metadata saved to: {meta_file_path}")
        
        return correct_labels, predicted_labels, predicted_scores, df_test, feature_file_name
    else:
        eva_dict, test_loss, correct_labels, predicted_labels, predicted_scores = Tester.test()
        print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")
        
        # 如果保存了特征，处理并保存（与上一个模型保持一致）
        feature_file_name = None
        if args.output_features and fusion_features_list and handle is not None:
            # 移除hook
            handle.remove()
            
            # 合并所有batch的特征
            all_features = np.vstack(fusion_features_list)
            
            # 确保特征数量与预测结果数量一致
            assert len(all_features) == len(predicted_scores), \
                f"Feature count ({len(all_features)}) doesn't match prediction count ({len(predicted_scores)})"
            
            # 创建输出目录（如果不存在）
            os.makedirs(args.output, exist_ok=True)
            
            # 保存特征为npy文件（二进制格式）
            feature_file_name = f"features_{args.test_name}.npy"
            feature_file_path = os.path.join(args.output, feature_file_name)
            np.save(feature_file_path, all_features)
            
            # # 保存特征元数据
            # meta_file_path = os.path.join(args.output, f"features_{args.test_name}_meta.json")
            # meta_data = {
            #     'test_name': args.test_name,
            #     'feature_file': feature_file_name,
            #     'num_samples': len(all_features),
            #     'feature_dim': all_features.shape[1] if len(all_features.shape) > 1 else 1,
            #     'dtype': str(all_features.dtype),
            #     'model_dir': args.model_dir,
            #     'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            # }
            # with open(meta_file_path, 'w') as f:
            #     json.dump(meta_data, f, indent=2)
            
            print(f"Fusion features saved to: {feature_file_path}")
            # print(f"Feature metadata saved to: {meta_file_path}")
        
        return eva_dict, correct_labels, predicted_labels, predicted_scores, df_test, feature_file_name


if __name__ == '__main__':
    s = time()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    if args.mode == 'predict':
        correct_labels, predicted_labels, predicted_scores, df_test, feature_file = main()
        
        # 构建预测结果字典
        predict_dict = {
            'Compound_ID': df_test["SMILES"] if "SMILES" in df_test.columns else range(len(predicted_scores)),
            'Protein_ID': df_test["Protein"] if "Protein" in df_test.columns else range(len(predicted_scores)),
            'Predicted_scores': predicted_scores, 
            'label_predict': predicted_labels
        }
        
        # 如果启用了特征输出，添加特征索引和文件名（与上一个模型保持一致）
        if args.output_features and feature_file is not None:
            predict_dict['feature_index'] = list(range(len(predicted_scores)))
            predict_dict['feature_file'] = feature_file
        
        output_file = f'{args.output}{args.test_name}_score.csv'
        print("save to %s" % output_file)
        result_df = pd.DataFrame(predict_dict)
        result_df.to_csv(output_file, index=False)
        
    else:
        eva_dict, correct_labels, predicted_labels, predicted_scores, df_test, feature_file = main()
        
        # 构建预测结果字典
        predict_dict = {
            'Compound_ID': df_test["SMILES"] if "SMILES" in df_test.columns else range(len(predicted_scores)),
            'Protein_ID': df_test["Protein"] if "Protein" in df_test.columns else range(len(predicted_scores)),
            'Predicted_scores': predicted_scores, 
            'label_predict': predicted_labels, 
            'label_original': df_test["Y"] if "Y" in df_test.columns else correct_labels,
            'correct_labels': correct_labels
        }
        
        # 如果启用了特征输出，添加特征索引和文件名（与上一个模型保持一致）
        if args.output_features and feature_file is not None:
            predict_dict['feature_index'] = list(range(len(predicted_scores)))
            predict_dict['feature_file'] = feature_file
        
        output_file = f'{args.output}{args.test_name}_score.csv'
        print("save to %s" % output_file)
        result_df = pd.DataFrame(predict_dict)
        result_df.to_csv(output_file, index=False)

        # 保存评估结果
        result_dic = {
            "test_name": [args.test_name], 
            "AUC": [eva_dict["AUC"]], 
            "AUPR": [eva_dict["PRC"]], 
            "ACC": [eva_dict["ACC"]], 
            "Rec": [eva_dict["Rec"]], 
            "Pre": [eva_dict["Pre"]], 
            "F1": [eva_dict["F1"]], 
            "MCC": [eva_dict["MCC"]]
        }
        df_eva = pd.DataFrame(result_dic)
        df_eva.to_csv(f'{args.output}evaluation.csv', index=False, mode='a')

    e = time()
    print(f"Total running time: {round(e - s, 2)}s")