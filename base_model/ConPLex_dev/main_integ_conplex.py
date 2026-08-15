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

from src.margin import MarginScheduledLossFunction

logg = get_logger()

parser = ArgumentParser(description="PLM_DTI Training.")

parser.add_argument(
    "--exp-id", required=True, help="Experiment ID", dest="experiment_id"
)
parser.add_argument("dataset_dir", help="dir of dataset [SMILES, sqeuence, label]")

# training_params
parser.add_argument("--lr", '-r', help="Learning late for training", default=1e-4, type=float)
parser.add_argument("--epochs", '-e', help="The number of epochs for training or validation", type=int, default=15)
parser.add_argument("--dropout", "-D", help="Dropout ratio", default=0.2, type=float)
parser.add_argument("--batch-size", "-b", help="Batch size", default=32, type=int)
parser.add_argument("--test", help="test", default=False, type=int)
# output_params
parser.add_argument("--output-model", "-m", help="save model", type=str)
parser.add_argument("--output", "-o", help="Prediction output", type=str)


parser.add_argument(
    "--config", "-c",  help="YAML config file", default="configs/default_config.yaml"
)

# parser.add_argument(
#     "--task",
#     choices=[
#         "biosnap",
#         "bindingdb",
#         "davis",
#         "biosnap_prot",
#         "biosnap_mol",
#         "dti_dg",
#     ],
#     type=str,
#     help="Task name. Could be biosnap, bindingdb, davis, biosnap_prot, biosnap_mol.",
# )

# parser.add_argument(
#     "--drug-featurizer", help="Drug featurizer", dest="drug_featurizer"
# )
# parser.add_argument(
#     "--target-featurizer", help="Target featurizer", dest="target_featurizer"
# )
# parser.add_argument(
#     "--distance-metric",
#     help="Distance in embedding space to supervise with",
#     dest="distance_metric",
# )
# parser.add_argument("--epochs", type=int, help="number of total epochs to run")
# parser.add_argument("-b", "--batch-size", type=int, help="batch size")
# parser.add_argument(
#     "--lr",
#     "--learning-rate",
#     type=float,
#     help="initial learning rate",
#     dest="lr",
# )
# parser.add_argument(
#     "--clr", type=float, help="initial learning rate", dest="clr"
# )
# parser.add_argument(
#     "--r", "--replicate", type=int, help="Replicate", dest="replicate"
# )
# parser.add_argument(
#     "--d", "--device", type=int, help="CUDA device", dest="device"
# )
# parser.add_argument(
#     "--verbosity", type=int, help="Level at which to log", dest="verbosity"
# )
# parser.add_argument(
#     "--checkpoint", default=None, help="Model weights to start from"
# )
def metrics_muti(correct_labels, predicted_labels, predicted_scores):
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

    metric_dict = {}

    for k, met_class in metrics.items():
        if classify:
            met_instance = met_class(task="binary")
        else:
            met_instance = met_class()
        met_instance.to(device)
        met_instance.reset()
        metric_dict[k] = met_instance

    model.eval()

    predicted_scores = []
    correct_labels = []
    results = {"AUC":[], "AUPR": [], "ACC":[], "Rec":[], "Pre":[], "F1":[], "MCC":[]}
    
    for i, batch in tqdm(enumerate(data_generator), total=len(data_generator)):

        pred, label, label_  = step(model, batch, device)
        pred_score = pred.to("cpu")
        label_la = label_.to("cpu")
        
        predicted_scores.extend(pred_score.tolist() if pred_score.dim() > 0 else [pred_score.item()])
        correct_labels.extend(label_la.tolist() if label_la.dim() > 0 else [label_la.item()])
        # predicted_scores.extend(pred_score.tolist())
        # correct_labels.extend(label_la.tolist())

        if classify:
            label = label.int()
        else:
            label = label.float()

        for _, met_instance in metric_dict.items():
            met_instance(pred, label)
    # print(predicted_scores)
    # print(correct_labels)
    predicted_labels = [1 if score >= 0.5 else 0 for score in predicted_scores]
    # print(predicted_labels)
    # print(type(correct_labels), type(predicted_labels), type(predicted_scores))
    ACC, AUC, Rec, Pre, F1, MCC, PRC = metrics_muti(correct_labels, predicted_labels, predicted_scores)
    # ACC, AUC, Rec, Pre, F1, MCC, PRC
    # results = {}
    for (k, met_instance) in metric_dict.items():
        res = met_instance.compute()
        results[k] = res

    for met_instance in metric_dict.values():
        met_instance.to("cpu")

    results["AUC"] = AUC
    results["AUPR"] = PRC
    results["ACC"] = ACC
    results["Rec"] = Rec
    results["Pre"] = Pre
    results["F1"] = F1
    results["MCC"] = MCC
    return results, AUC, PRC, ACC, Rec, Pre, F1, MCC


def step(model, batch, device=None):

    if device is None:
        device = torch.device("cpu")
    # print(device)
    drug, target, label_ = batch  # target is (D + N_pool)
    # print(drug.size(), target.size(), label_.size())
    pred = model(drug.to(device), target.to(device))
    # print(pred)
    label = Variable(torch.from_numpy(np.array(label_)).float()).to(device)
    return pred, label, label_ 


def contrastive_step(model, batch, device=None):

    if device is None:
        device = torch.device("cpu")

    anchor, positive, negative = batch

    anchor_projection = model.target_projector(anchor.to(device))
    positive_projection = model.drug_projector(positive.to(device))
    negative_projection = model.drug_projector(negative.to(device))

    return anchor_projection, positive_projection, negative_projection




def main():
    # Get configuration
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(arg_overrides)

    # save_dir = f'{config.get("model_save_dir", ".")}/{config.experiment_id}'
    # os.makedirs(save_dir, exist_ok=True)

    # Logging
    if "log_file" not in config:
        config.log_file = None
    else:
        os.makedirs(Path(config.log_file).parent, exist_ok=True)
    config_logger(
        config.log_file,
        "%(asctime)s [%(levelname)s] %(message)s",
        config.verbosity,
        use_stdout=True,
    )

    # Set CUDA device
    device_no = config.device
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    logg.info(f"Using CUDA device {device}")

    # Set random state
    logg.debug(f"Setting random state {config.replicate}")
    set_random_seed(config.replicate)

    # Load DataModule
    logg.info("Preparing DataModule")
    # task_dir = get_task_dir(config.task)

    task_dir = config.dataset_dir

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
    datamodule.prepare_data()
    datamodule.setup()

    # Load DataLoaders
    logg.info("Getting DataLoaders")
    training_generator = datamodule.train_dataloader()
    validation_generator = datamodule.val_dataloader()
    testing_generator = datamodule.test_dataloader()

    if config.contrastive:
        logg.info("Loading contrastive data (DUDE)")
        dude_drug_featurizer = get_featurizer(
            config.drug_featurizer, save_dir=get_task_dir("DUDe")
        )

        dude_target_featurizer = get_featurizer(
            config.target_featurizer, save_dir=get_task_dir("DUDe")
        )

        contrastive_datamodule = DUDEDataModule(
            config.contrastive_split,
            dude_drug_featurizer,
            dude_target_featurizer,
            device=device,
            batch_size=config.contrastive_batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )

        contrastive_datamodule.prepare_data()
        contrastive_datamodule.setup(stage="fit")
        contrastive_generator = contrastive_datamodule.train_dataloader()

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
    if "checkpoint" in config:
        state_dict = torch.load(config.checkpoint)
        model.load_state_dict(state_dict)

    model = model.to(device)
    logg.info(model)

    # Optimizers
    logg.info("Initializing optimizers")
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=config.lr_t0
    )

    if config.contrastive:
        contrastive_loss_fct = MarginScheduledLossFunction(
            M_0=config.margin_max,
            N_epoch=config.epochs,
            N_restart=config.margin_t0,
            update_fn=config.margin_fn,
        )
        opt_contrastive = torch.optim.AdamW(model.parameters(), lr=config.clr)
        lr_scheduler_contrastive = (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_contrastive, T_0=config.clr_t0
            )
        )

    # Metrics
    logg.info("Initializing metrics")
    max_metric = 0
    model_max = copy.deepcopy(model)

    if config.task == "dti_dg":
        loss_fct = torch.nn.MSELoss()
        val_metrics = {
            "val/mse": torchmetrics.MeanSquaredError,
            "val/pcc": torchmetrics.PearsonCorrCoef,
        }

        test_metrics = {
            "test/mse": torchmetrics.MeanSquaredError,
            "test/pcc": torchmetrics.PearsonCorrCoef,
        }
    else:
        # loss_fct = torch.nn.BCELoss()
        loss_fct = torch.nn.BCEWithLogitsLoss()
        val_metrics = {
            "val/aupr": torchmetrics.AveragePrecision,
            "val/auroc": torchmetrics.AUROC,
        }

        test_metrics = {
            "test/aupr": torchmetrics.AveragePrecision,
            "test/auroc": torchmetrics.AUROC,
        }

    logg.info("Config:")
    logg.info(json.dumps(dict(config), indent=4))

    logg.info("Beginning Training")

    torch.backends.cudnn.benchmark = True

    # Begin Training
    start_time = time()
    result_dict = {"learning_rate": [], "batch_size": [], "dropout": [], "n_epoch": [], "AUC": [], "AUPR": [], "ACC": [], "Rec": [], "Pre": [], "F1": [], "MCC": [], "AUC_test": [], "AUPR_test": [], "ACC_test": [], "Rec_test": [], "Pre_test": [], "F1_test": [], "MCC_test": []}

    for epo in range(config.epochs):
        model.train()
        epoch_time_start = time()

        # Main Step
        for i, batch in tqdm(
            enumerate(training_generator), total=len(training_generator)
        ):
            pred, label, label_ = step(
                model, batch, device
            )  # batch is (2048, 1024, 1)
            # print('pred:',pred)
            # print('label:',label)
            # print('label:',label_)
            loss = loss_fct(pred, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

        lr_scheduler.step()

        logg.info(
            f"Training at Epoch {epo + 1} with loss {loss.cpu().detach().numpy():8f}"
        )
        logg.info(f"Updating learning rate to {lr_scheduler.get_lr()[0]:8f}")

        # Contrastive Step
        if config.contrastive:
            logg.info(f"Training contrastive at Epoch {epo + 1}")
            for i, batch in tqdm(
                enumerate(contrastive_generator),
                total=len(contrastive_generator),
            ):

                anchor, positive, negative = contrastive_step(
                    model, batch, device
                )

                contrastive_loss = contrastive_loss_fct(
                    anchor, positive, negative
                )

                opt_contrastive.zero_grad()
                contrastive_loss.backward()
                opt_contrastive.step()

            contrastive_loss_fct.step()
            lr_scheduler_contrastive.step()

            logg.info(
                f"Training at Contrastive Epoch {epo + 1} with loss {contrastive_loss.cpu().detach().numpy():8f}"
            )
            logg.info(
                f"Updating contrastive learning rate to {lr_scheduler_contrastive.get_lr()[0]:8f}"
            )
            logg.info(
                f"Updating contrastive margin to {contrastive_loss_fct.margin}"
            )

        epoch_time_end = time()

        # Validation
        if epo % config.every_n_val == 0:
            with torch.set_grad_enabled(False):

                model_save_path = config.output_model
                torch.save(
                    model_max.state_dict(),
                    f'{model_save_path[:-6]}_epo{epo}.model',
                )
                logg.info(f"Saving checkpoint model to {model_save_path}")
                # ACC, AUC, Rec, Pre, F1, MCC, PRC
                val_results, AUC, PRC, ACC, Rec, Pre, F1, MCC = test(
                    model,
                    validation_generator,
                    val_metrics,
                    device,
                    config.classify,
                )

                epoch_time = time() - epoch_time_start
                print('Time/epoch_time', epoch_time, epo)

                epoch_start_predict = time()
                test_results, AUC_test, PRC_test, ACC_test, Rec_test, Pre_test, F1_test, MCC_test = test(
                    model,
                    testing_generator,
                    test_metrics,
                    device,
                    config.classify,
                )
                epoch_time_predict = time() - epoch_start_predict
                print('Time/predict_time', epoch_time_predict, epo + 1)


                val_results["epoch"] = epo + 1
                val_results["Charts/epoch_time"] = (
                    epoch_time_end - epoch_time_start
                ) / config.every_n_val


                if AUC > max_metric:
                    # logg.debug(
                    #     f"Validation AUPR {val_results[config.watch_metric]:8f} > previous max {max_metric:8f}"
                    # )
                    model_max = copy.deepcopy(model)
                    max_metric = AUC
                    # model_save_path = Path(
                    #     f"{save_dir}/{config.experiment_id}_best_model_epoch{epo:02}.pt"
                    # )
                    model_save_path = config.output_model
                    torch.save(
                        model_max.state_dict(),
                        f'{model_save_path[:-6]}_best.model',
                    )
                    logg.info(f"Saving checkpoint model to {model_save_path}")

                logg.info(f"Validation at Epoch {epo + 1}")
                for k, v in val_results.items():
                    if not k.startswith("_"):
                        logg.info(f"{k}: {v}")

        
        result_dict ["learning_rate"].append(config.lr)
        result_dict ["batch_size"].append(config.batch_size)
        result_dict ["dropout"].append(config.dropout)
        result_dict ["n_epoch"].append(epo + 1)
        # result_dict ["AUC"].append(AUC)
        # result_dict ["AUPR"].append(PRC)
        # result_dict ["val/auroc"] = val_results["val/auroc"]
        # result_dict ["val/aupr"] = val_results["val/aupr"]
        result_dict ["ACC"].append(ACC)
        result_dict ["Rec"].append(Rec)
        result_dict ["Pre"].append(Pre)
        result_dict ["F1"].append(F1)
        result_dict ["MCC"].append(MCC)
        result_dict ["AUC"].append(AUC)
        result_dict ["AUPR"].append(PRC)
        # result_dict ["val/auroc"] = val_results["val/auroc"]
        # result_dict ["val/aupr"] = val_results["val/aupr"]
        result_dict ["ACC_test"].append(ACC_test)
        result_dict ["Rec_test"].append(Rec_test)
        result_dict ["Pre_test"].append(Pre_test)
        result_dict ["F1_test"].append(F1_test)
        result_dict ["MCC_test"].append(MCC_test)
        result_dict["AUC_test"].append(AUC_test if AUC_test is not None else 0.0)
        result_dict["AUPR_test"].append(PRC_test if PRC_test is not None else 0.0)

    print(result_dict)
    result_df = pd.DataFrame(result_dict)
    print("save to " + config.output)
    print(result_df)
    result_df.to_csv(config.output, index=False, mode='a')
    end_time = time()

    # Testing
    if config.test:
        logg.info("Beginning testing")
        try:
            with torch.set_grad_enabled(False):
                model_max = model_max.eval()

                test_start_time = time()
                test_results, AUC, PRC, ACC, Rec, Pre, F1, MCC = test(
                    model_max,
                    testing_generator,
                    test_metrics,
                    device,
                    config.classify,
                )
                test_end_time = time()

                test_results["epoch"] = epo + 1
                test_results["test/eval_time"] = test_end_time - test_start_time
                test_results["Charts/wall_clock_time"] = end_time - start_time

                logg.info("Final Testing")
                for k, v in test_results.items():
                    if not k.startswith("_"):
                        logg.info(f"{k}: {v}")

                model_save_path = Path(
                    f"{save_dir}/{config.experiment_id}_best_model.pt"
                )
                torch.save(
                    model_max.state_dict(),
                    model_save_path,
                )
                logg.info(f"Saving final model to {model_save_path}")

        except Exception as e:
            logg.error(f"Testing failed with exception {e}")

    return model_max


best_model = main()
