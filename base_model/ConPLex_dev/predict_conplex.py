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
parser.add_argument(
    "--config", help="YAML config file", default="configs/default_config.yaml"
)

# nohup python predict_complex.py ./output/$data_dir/models/$data_sub/model_$data_dir-$learning_rate-$batch_size-${dropout}_best.model test ${data_dir}_$data_sub ./conplex_cpi_data/$data_dir/$data_sub/test.csv -o ./output/$data_dir/ > ./output/$data_dir/myout_${data_dir}_$data_sub.file 2>&1 &    

# nohup python predict_complex.py ./output/$data_dir/models/model_$data_dir-$learning_rate-$batch_size-${dropout}_best.model test ${data_dir}_$data_sub ./conplex_cpi_data/$data_dir/$data_sub.csv -o ./output/$data_dir/ > ./output/$data_dir/myout_${data_dir}_$data_sub.file 2>&1 &    

def metrics(correct_labels, predicted_labels, predicted_scores):
    ACC = accuracy_score(correct_labels, predicted_labels)
    AUC = roc_auc_score(correct_labels, predicted_scores)
    CM = confusion_matrix(correct_labels, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Rec = TP / (TP + FN)
    Pre = TP / (TP + FP)
    F1 = 2 * Pre * Rec / (Pre + Rec)
    MCC = matthews_corrcoef(correct_labels, predicted_labels)
    precision, recall, _ = precision_recall_curve(correct_labels, predicted_scores)
    PRC = auc(recall, precision)
    return ACC, AUC, Rec, Pre, F1, MCC, PRC


def test(model, data_generator, metrics, device=None, classify=True):

    if device is None:
        device = torch.device("cpu")


    model.eval()
    # smiles_list = []
    # sequences_list = []
    labels_origin = []
    predicted_scores = []
    correct_labels = []
    smiles_list = []
    sequences_list = []

    for i, batch in tqdm(enumerate(data_generator), total=len(data_generator)):

        pred, label, label_, smiles, sequences = step(model, batch, device)
        print(pred, label_, smiles, sequences)

        # print(smiles.tolist())
        # print(sequences.tolist())
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



    predicted_labels = [1 if x >= 0.5 else 0 for x in predicted_scores]
    # AUC, PRC, ACC, Rec, Pre, F1, MCC = metrics(correct_labels, predicted_labels, predicted_scores)
    # print(AUC, PRC, ACC, Rec, Pre, F1, MCC)

    # results_ev = {"AUC":[AUC], "AUPR": [PRC], "ACC":[ACC], "Rec":[Rec], "Pre":[Pre], "F1":[F1], "MCC":[MCC]}
    # results_ev["AUC"] = AUC
    # results_ev["PRC"] = PRC
    # results_ev["ACC"] = ACC
    # results_ev["Rec"] = Rec
    # results_ev["Pre"] = Pre
    # results_ev["F1"] = F1
    # results_ev["MCC"] = MCC

    print('predicted_scores:', predicted_scores)
    print('predicted_labels:', predicted_labels)
    print('smiles_list:', smiles_list)
    print('sequences_list:', sequences_list)
    print('labels_origin:', correct_labels)

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
    # os.makedirs(save_dir, exist_ok=True)

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
            test_path=f"{task_dir}/test.csv",    # 新增参数
        )
        print('datamodule: DTIDataModule')
    datamodule.prepare_data_test()
    print('data_test prepared')
    index_to_smiles_test, index_to_sequences_test, df_test = datamodule.setup_test()
    print('data_test setuped')
    print(df_test)

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
                )

        except Exception as e:
            logg.error(f"Testing failed with exception {e}")

        # smiles_list_origin = [index_to_smiles_test[idx.item()] for idx in smiles_list]
        # sequences_list_origin = [index_to_sequences_test[idx.item()] for idx in sequences_list]

        if config.mode == 'predict':
            predict_dict = {'Compound_ID': df_test["SMILES"], 'Protein_ID': df_test["Target Sequence"], 'Predicted_scores': predicted_scores, 'label_predict': predicted_labels}
            print("save to %s"%f'{config.output}{args.test_name}_score.csv')
            result_df = pd.DataFrame(predict_dict)
            result_df.to_csv(f'{config.output}{args.test_name}_score.csv', index=False)
            # AUC, PRC, ACC, Rec, Pre, F1, MCC = metrics(result_df['label_original'], result_df['label_predict'], result_df['Predicted_scores'])
        else:
            predict_dict = {'Compound_ID': df_test["SMILES"], 'Protein_ID': df_test["Target Sequence"], 'Predicted_scores': predicted_scores, 'label_predict': predicted_labels, 'label_original': df_test["Label"]}
            print("save to %s"%f'{config.output}{args.test_name}_score.csv')
            result_df = pd.DataFrame(predict_dict)
            result_df.to_csv(f'{config.output}{args.test_name}_score.csv', index=False)
            ACC, AUC, Rec, Pre, F1, MCC, PRC = metrics(result_df['label_original'], result_df['label_predict'], result_df['Predicted_scores'])
            
            print(AUC, PRC, ACC, Rec, Pre, F1, MCC)

            result_dic = {"test_name":args.test_name, "AUC":[AUC] , "AUPR": [PRC] , "ACC":[ACC] , "Rec":[Rec] , "Pre":[Pre] , "F1":[F1] , "MCC":[MCC] }
            df_eva = pd.DataFrame(result_dic)
            df_eva.to_csv(f'{config.output}evaluation.csv', index=False, mode='a')

best_model = main()
