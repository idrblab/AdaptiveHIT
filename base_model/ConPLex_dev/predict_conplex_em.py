import copy
from time import time
import os
import sys
import numpy as np
import pandas as pd
import torch
import json
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import typing as T
import torchmetrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef, precision_recall_curve, auc

from argparse import ArgumentParser

from omegaconf import OmegaConf
from pathlib import Path

from src import architectures as model_types
from src.data import (
    get_task_dir,
    DTIDataModule,
    TDCDataModule,
    DUDEDataModule,
    EnzPredDataModule,
)
from src.utils import (
    set_random_seed,
    config_logger,
    get_logger,
    get_featurizer,
    sigmoid_cosine_distance_p,
)

logg = get_logger()
parser = ArgumentParser(description="PLM_DTI Training.")
parser.add_argument(
    "--exp-id", required=True, help="Experiment ID", dest="experiment_id"
)
# training_params batch_size 
parser.add_argument("batch_size", help="batch_size of model",type=int)
parser.add_argument("model_dir", help="dir of model ")
parser.add_argument("mode", help="mode of model")
parser.add_argument("test_name", help="name of test ")
parser.add_argument("dataset_test_dir", help="dir of dataset_test [SMILES, sqeuence, label]")

# output_params
parser.add_argument("--output", "-o", help="Prediction output", type=str)
# 新增参数：是否输出融合特征（与之前模型保持一致）
parser.add_argument("--output_features", action="store_true", 
                    help="Output fused features before classifier as binary files")
parser.add_argument(
    "--config", help="YAML config file", default="configs/default_config.yaml"
)

# 定义hook函数来捕获融合特征
fusion_features_list = []

def hook_fn(module, input, output):
    """
    Hook function to capture fusion features
    For SimpleCoembedding models, the fusion features are the projected drug and target features
    We'll capture them before the distance calculation
    """
    fusion_features_list.append(output.detach().cpu().numpy())

def test(model, data_generator, metrics, device=None, classify=True, output_features=False):

    if device is None:
        device = torch.device("cpu")

    model.eval()
    labels_origin = []
    predicted_scores = []
    correct_labels = []
    smiles_list = []
    sequences_list = []
    
    # 如果需要输出特征，注册hook
    handle = None
    if output_features:
        # 对于SimpleCoembedding模型，我们想要捕获融合后的特征
        # 根据模型架构，我们需要找到合适的层来注册hook
        # 如果是SimpleCoembedding_FoldSeek，可以hook target_projector的输出
        if hasattr(model, 'target_projector') and callable(model.target_projector):
            # 对于有target_projector方法的模型
            handle = model.target_projector.register_forward_hook(hook_fn)
            print("Registered hook on target_projector")
        elif hasattr(model, '_target_projector'):
            # 对于SimpleCoembedding_FoldSeekX
            handle = model._target_projector.register_forward_hook(hook_fn)
            print("Registered hook on _target_projector")
        elif hasattr(model, 'drug_projector') and hasattr(model, 'target_projector'):
            # 对于简单模型，我们可以hook drug_projector的输出
            # 但实际上我们想要的是投影后的特征，可以hook整个模型的前向传播
            handle = model.register_forward_hook(hook_fn)
            print("Registered hook on model")
        else:
            print("Warning: Could not find suitable layer for feature extraction")

    for i, batch in tqdm(enumerate(data_generator), total=len(data_generator)):
        pred, label, label_, smiles, sequences = step(model, batch, device)

        num_elements = pred.numel()
        if num_elements == 1:
            pred = pred.item()
            label_ = label_.item()
            smiles = smiles.item()
            sequences = sequences.item()

            predicted_scores.append(pred)
            correct_labels.append(label_)
            smiles_list.append(smiles)
            sequences_list.append(sequences)
            labels_origin.append(label_)
        else:
            predicted_scores.extend(pred.tolist())
            correct_labels.extend(label_.tolist())
            smiles_list.extend(smiles.tolist())
            sequences_list.extend(sequences.tolist())
            labels_origin.extend(label_.tolist())

    # 移除hook
    if handle is not None:
        handle.remove()

    predicted_labels = [1 if x >= 0.5 else 0 for x in predicted_scores]

    print('predicted_scores:', predicted_scores[:5])
    print('predicted_labels:', predicted_labels[:5])
    print('smiles_list:', smiles_list[:5])
    print('sequences_list:', sequences_list[:5])
    print('labels_origin:', correct_labels[:5])

    return predicted_scores, predicted_labels, smiles_list, sequences_list, labels_origin


def step(model, batch, device=None):

    if device is None:
        device = torch.device("cpu")

    drug, target, label_, smiles, sequence = batch  # target is (D + N_pool)
    pred = model(drug.to(device), target.to(device))
    label = Variable(torch.from_numpy(np.array(label_)).float()).to(device)
    return pred, label, label_, smiles, sequence


def main():
    # Get configuration
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(arg_overrides)

    save_dir = f'{config.get("model_save_dir", ".")}/{config.experiment_id}'

    # Logging
    if "log_file" not in config:
        config.log_file = None
    else:
        os.makedirs(Path(config.log_file).parent, exist_ok=True)
        print('log_file')
    config_logger(
        config.log_file,
        "%(asctime)s [%(levelname)s] %(message)s",
        config.verbosity,
        use_stdout=True,
    )

    # Set CUDA device
    device_no = config.device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    logg.info(f"Using CUDA device {device}")

    # Set random state
    logg.debug(f"Setting random state {config.replicate}")
    set_random_seed(config.replicate)


    task_dir = config.dataset_test_dir

    drug_featurizer = get_featurizer(config.drug_featurizer, save_dir=task_dir)
    target_featurizer = get_featurizer(
        config.target_featurizer, save_dir=task_dir
    )

    if config.task == "dti_dg":
        config.classify = False
        config.watch_metric = "val/pcc"
        datamodule = TDCDataModule(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            seed=config.replicate,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
        print('datamodule: TDCDataModule')
    elif config.task in EnzPredDataModule.dataset_list():
        config.classify = True
        config.watch_metric = "val/aupr"
        datamodule = EnzPredDataModule(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            seed=config.replicate,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
        print('datamodule: EnzPredDataModule')
    else:
        config.classify = True
        config.watch_metric = "val/aupr"
        datamodule = DTIDataModule(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
        print('datamodule: DTIDataModule')
    datamodule.prepare_data_test()
    print('data_test prepared')
    index_to_smiles_test, index_to_sequences_test, df_test = datamodule.setup_test()
    print('data_test setuped')
    print(df_test.head())

    # Load DataLoaders
    logg.info("Getting DataLoaders")

    testing_generator = datamodule.test_dataloader_test()

    config.drug_shape = drug_featurizer.shape
    config.target_shape = target_featurizer.shape

    # Model
    logg.info("Initializing model")
    model = getattr(model_types, config.model_architecture)(
        config.drug_shape,
        config.target_shape,
        latent_dimension=config.latent_dimension,
        latent_distance=config.latent_distance,
        classify=config.classify,
    )

    state_dict = torch.load(config.model_dir, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print('model load success')

    # Metrics
    model_max = copy.deepcopy(model)

    if config.task == "dti_dg":
        test_metrics = {
            "test/mse": torchmetrics.MeanSquaredError,
            "test/pcc": torchmetrics.PearsonCorrCoef,
        }
    else:
        test_metrics = {
            "test/aupr": torchmetrics.AveragePrecision,
            "test/auroc": torchmetrics.AUROC,
        }

    torch.backends.cudnn.benchmark = True

    # Testing
    if config.mode == 'test' or config.mode == 'predict':
        try:
            with torch.set_grad_enabled(False):
                model_max = model_max.eval()
                predicted_scores, predicted_labels, smiles_list, sequences_list, labels_origin = test(
                    model_max,
                    testing_generator,
                    test_metrics,
                    device,
                    config.classify,
                    output_features=args.output_features,
                )

        except Exception as e:
            logg.error(f"Testing failed with exception {e}")

        # 如果保存了特征，处理并保存（与之前模型保持一致）
        feature_file_name = None
        if args.output_features and fusion_features_list:
            # 合并所有batch的特征
            all_features = np.vstack(fusion_features_list)
            
            # 确保特征数量与预测结果数量一致
            assert len(all_features) == len(predicted_scores), \
                f"Feature count ({len(all_features)}) doesn't match prediction count ({len(predicted_scores)})"
            
            # 创建输出目录（如果不存在）
            os.makedirs(os.path.dirname(config.output), exist_ok=True)
            
            # 保存特征为npy文件（二进制格式）
            feature_file_name = f"features_{args.test_name}.npy"
            feature_file_path = os.path.join(os.path.dirname(config.output), feature_file_name)
            np.save(feature_file_path, all_features)
            
            # # 保存特征元数据
            # meta_file_path = os.path.join(os.path.dirname(config.output), f"features_{args.test_name}_meta.json")
            # meta_data = {
            #     'test_name': args.test_name,
            #     'feature_file': feature_file_name,
            #     'num_samples': len(all_features),
            #     'feature_dim': all_features.shape[1] if len(all_features.shape) > 1 else 1,
            #     'dtype': str(all_features.dtype),
            #     'model_dir': config.model_dir,
            #     'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            # }
            # with open(meta_file_path, 'w') as f:
            #     json.dump(meta_data, f, indent=2)
            
            print(f"Fusion features saved to: {feature_file_path}")
            # print(f"Feature metadata saved to: {meta_file_path}")

        if config.mode == 'predict':
            predict_dict = {
                'Compound_ID': df_test["SMILES"] if "SMILES" in df_test.columns else smiles_list,
                'Protein_ID': df_test["Target Sequence"] if "Target Sequence" in df_test.columns else sequences_list,
                'Predicted_scores': predicted_scores, 
                'label_predict': predicted_labels
            }
            
            # 如果启用了特征输出，添加特征索引和文件名
            if args.output_features and feature_file_name is not None:
                predict_dict['feature_index'] = list(range(len(predicted_scores)))
                predict_dict['feature_file'] = feature_file_name
            
            print("save to %s" % f'{config.output}{args.test_name}_score.csv')
            result_df = pd.DataFrame(predict_dict)
            result_df.to_csv(f'{config.output}{args.test_name}_score.csv', index=False)
        else:
            predict_dict = {
                'Compound_ID': df_test["SMILES"] if "SMILES" in df_test.columns else smiles_list,
                'Protein_ID': df_test["Target Sequence"] if "Target Sequence" in df_test.columns else sequences_list,
                'Predicted_scores': predicted_scores, 
                'label_predict': predicted_labels, 
                'label_original': df_test["Label"] if "Label" in df_test.columns else labels_origin
            }
            
            # 如果启用了特征输出，添加特征索引和文件名
            if args.output_features and feature_file_name is not None:
                predict_dict['feature_index'] = list(range(len(predicted_scores)))
                predict_dict['feature_file'] = feature_file_name
            
            print("save to %s" % f'{config.output}{args.test_name}_score.csv')
            result_df = pd.DataFrame(predict_dict)
            result_df.to_csv(f'{config.output}{args.test_name}_score.csv', index=False)
            
            ACC, AUC, Rec, Pre, F1, MCC, PRC = metrics(
                result_df['label_original'], 
                result_df['label_predict'], 
                result_df['Predicted_scores']
            )
            
            print(f"AUC: {AUC}, PRC: {PRC}, ACC: {ACC}, Rec: {Rec}, Pre: {Pre}, F1: {F1}, MCC: {MCC}")

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
            df_eva.to_csv(f'{config.output}evaluation.csv', index=False, mode='a')

if __name__ == "__main__":
    main()