import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model
import argparse
import h5py
from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef


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
    Rec = TP / (TP + FN)
    Pre = TP / (TP + FP)
    F1 = 2 * Pre * Rec / (Pre + Rec)
    MCC = matthews_corrcoef(test_label, predicted_labels)
    precision, recall, _ = precision_recall_curve(test_label, prediction)
    PRC = auc(recall, precision)
    return ACC, AUC, Rec, Pre, F1, MCC, PRC

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0] 
    else:
        return [seq_dic[aa] for aa in seq]

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [seq_dic[aa] for aa in seq]

def parse_data(dti_dir, drug_dir, protein_dir, with_label=True,
               prot_len=2500, prot_vec="Convolution",
               drug_vec="Convolution", drug_len=2048):

    print("Parsing {0} , {1}, {2} with length {3}, type {4}".format(*[dti_dir ,drug_dir, protein_dir, prot_len, prot_vec]))

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
        return {"protein_feature": protein_feature, "drug_feature": drug_feature, "label": label,
                "Compound_ID":dti_df["SMILES"].tolist(), "Protein_ID":dti_df["Sequence"].tolist()}
    else:
        return {"protein_feature": protein_feature, "drug_feature": drug_feature,
                "Compound_ID":dti_df["SMILES"].tolist(), "Protein_ID":dti_df["Sequence"].tolist()}



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--mode",'-m', help="mode of model", default='test' ,type=str)
    parser.add_argument("--with-label", "-W", help="Existence of label information in test DTI", default='N',type=str)
    # test_params
    parser.add_argument("--test-name", '-n', help="Name of test data sets", type=str)
    parser.add_argument("--test-dti-dir", "-i", help="Test dti [drug, target, [label]]", nargs="*")
    parser.add_argument("--test-drug-dir", "-d", help="Test drug information [drug, SMILES,[feature_name, ..]]", nargs="*")
    parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]", nargs="*")
    parser.add_argument("--output", "-o", help="Prediction output", type=str)


    parser.add_argument("--prot-vec", "-v", help="Type of protein feature, if Convolution, it will execute conlvolution on sequeunce", type=str, default="Convolution")
    parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)
    parser.add_argument("--drug-vec", "-V", help="Type of drug feature", type=str, default="morgan_fp")
    parser.add_argument("--drug-len", "-L", help="Drug vector length", default=2048, type=int)
    args = parser.parse_args()
    
    model = args.model
    test_names = args.test_name
    tests = args.test_dti_dir
    test_proteins = args.test_protein_dir
    test_drugs = args.test_drug_dir
    test_sets = zip(test_names, tests, test_drugs, test_proteins)
    mode = args.mode
    output_file = args.output
    with_label = args.with_label
    if with_label =='T':
        with_label = True
    else:
        with_label = False

    f = h5py.File(model, 'r+')

    try:
        f.__delitem__("optimizer_weights")
    except:
        print("optimizer_weights are already deleted")

    f.close()

    type_params = {
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
        "drug_vec": args.drug_vec,
        "drug_len": args.drug_len,
    }
    test_dic = {test_name: parse_data(test_dti, test_drug, test_protein, with_label=with_label, **type_params)
                for test_name, test_dti, test_drug, test_protein in test_sets}
    if tf.test.is_gpu_available():
        with tf.device('/gpu:0'):
            loaded_model = load_model(model)
    else:
        with tf.device('/cpu:0'):
            loaded_model = load_model(model)
    print("prediction")
    for dataset in test_dic:
        prediction_dic = test_dic[dataset]
        N = int(np.ceil(prediction_dic["drug_feature"].shape[0]/50))
        d_splitted = np.array_split(prediction_dic["drug_feature"], N)
        p_splitted = np.array_split(prediction_dic["protein_feature"], N)
        predicted = sum([np.squeeze(loaded_model.predict([d,p])).tolist() for d,p in zip(d_splitted, p_splitted)], [])
        if mode == 'predict':
            predict_dict = {'Compound_ID': prediction_dic["Compound_ID"], 'Protein_ID': prediction_dic["Protein_ID"], 'Predicted_scores': predicted, 'label_predict': [1 if x >= 0.5 else 0 for x in predicted]}
            print("save to %s"%f'{output_file}{args.test_name}_score.csv')
            result_df = pd.DataFrame(predict_dict)
            result_df.to_csv(f'{output_file}{args.test_name}_score.csv', index=False)
            
        else:
            label_original = np.squeeze(test_dic[dataset]['label'])
            predict_dict = {'Compound_ID': prediction_dic["Compound_ID"], 'Protein_ID': prediction_dic["Protein_ID"], 'Predicted_scores': predicted, 'label_predict': [1 if x >= 0.5 else 0 for x in predicted], 'label_original': label_original}
            print("save to %s"%f'{output_file}{args.test_name}_score.csv')
            result_df = pd.DataFrame(predict_dict)
            result_df.to_csv(f'{output_file}{args.test_name}_score.csv', index=False)
            
            ACC, AUC, Rec, Pre, F1, MCC, PRC = evaluation(predict_dict['label_original'], predict_dict['label_predict'], predict_dict['Predicted_scores'])
            result_dic = {"test_name":[args.test_name], "AUC":[AUC], "AUPR": [PRC], "ACC":[ACC], "Rec":[Rec], "Pre":[Pre], "F1":[F1], "MCC":[MCC]}
            df_eva = pd.DataFrame(result_dic)
            df_eva.to_csv(f'{output_file}evaluation.csv', index=False, mode='a')

# nohup python predict_with_model.py model.py test -n bindingdb/test -i ./deepconv_cpi_data/$data_dir/dev/dti.csv  -d ./deepconv_cpi_data/$data_dir/dev/compound.csv -t ./deepconv_cpi_data/$data_dir/dev/protein.csv -o ./output/$data_dir/output -v Convolution -l 2500 -V morgan_fp_r2 -L 2048
