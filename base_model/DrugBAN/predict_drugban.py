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


args = parser.parse_args()
batch_size = args.batch_size
dataset_dir = args.dataset_test_dir
# nohup python predict_drugban.py batch_size model_dir test test_name ./drugban_cpi_data/human_cold/cold_1 -o output  > ./output/human_cold/myouts/cold_1/myout_human_cold-0.001-32.file 2>&1 &
# nohup python main_new.py ./drugban_cpi_data/human_cold/cold_1 -r 0.001 -e 1 -b 32 -m ./output/human_cold/models/cold_1/model_human_cold-0.001-32.model -o ./output/human_cold/results/cold_1/validation_output__human_cold.csv > ./output/human_cold/myouts/cold_1/myout_human_cold-0.001-32.file 2>&1 &
def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.cfg)
    cfg.SOLVER.BATCH_SIZE = batch_size
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]

    # print(f"Config yaml: {hum}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    # test_path = dataset_dir
    # dataFolder = os.path.join(dataFolder, str(args.split))

    if not cfg.DA.TASK:
        # test_path = os.path.join(dataFolder, "test.csv")
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


    if not cfg.DA.USE:
        Tester = Predicter(model, device, test_generator, **cfg)
    else:
        Tester = Predicter(model, device, test_generator, **cfg)
    if args.mode == 'predict':    
        test_loss, correct_labels, predicted_labels, predicted_scores = Tester.predict()
        print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")
        return correct_labels, predicted_labels, predicted_scores, df_test
    else:
        eva_dict, test_loss, correct_labels, predicted_labels, predicted_scores = Tester.test()
        print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")
        return eva_dict, correct_labels, predicted_labels, predicted_scores, df_test
    # with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
    #     wf.write(str(model))



if __name__ == '__main__':
    s = time()
    # eva_dict, correct_labels, predicted_labels, predicted_scores, df_test = main()


    if args.mode == 'predict':
        correct_labels, predicted_labels, predicted_scores, df_test = main()
        predict_dict = {'Compound_ID': df_test["SMILES"], 'Protein_ID': df_test["Protein"], 'Predicted_scores': predicted_scores, 'label_predict': predicted_labels}
        print("save to %s"%f'{args.output}{args.test_name}_score.csv')
        result_df = pd.DataFrame(predict_dict)
        result_df.to_csv(f'{args.output}{args.test_name}_score.csv', index=False)
    else:
        eva_dict, correct_labels, predicted_labels, predicted_scores, df_test = main()
        predict_dict = {'Compound_ID': df_test["SMILES"], 'Protein_ID': df_test["Protein"], 'Predicted_scores': predicted_scores, 'label_predict': predicted_labels, 'label_original': df_test["Y"], 'correct_labels': correct_labels}
        print("save to %s"%f'{args.output}{args.test_name}_score.csv')
        result_df = pd.DataFrame(predict_dict)
        result_df.to_csv(f'{args.output}{args.test_name}_score.csv', index=False)


        result_dic = {"test_name":args.test_name, "AUC":eva_dict["AUC"] , "AUPR": eva_dict["PRC"] , "ACC":eva_dict["ACC"] , "Rec":eva_dict["Rec"] , "Pre":eva_dict["Pre"] , "F1":eva_dict["F1"] , "MCC":eva_dict["MCC"] }
        df_eva = pd.DataFrame(result_dic)
        df_eva.to_csv(f'{args.output}evaluation.csv', index=False, mode='a')

    e = time()
    print(f"Total running time: {round(e - s, 2)}s")