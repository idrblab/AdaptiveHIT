import os
import sys
import pandas as pd

def data_file_dir_csv(folder_path):
    csv_files = [file[:-4] for file in os.listdir(folder_path) if file.endswith('.csv')]
    return csv_files

def ave_vote_data(df_trans, mode, models_mode, output_dir, file_name):
    for index, row in df_trans.iterrows():
        if mode == 4:
            pred_sum = (row['label_predict_DeepConv-DTI'] + row['label_predict_ConPLex'] + 
                       row['label_predict_DrugBAN'] + row['label_predict_TransformerCPI'])
            avg_score = (row['Predicted_scores_DeepConv-DTI'] + row['Predicted_scores_ConPLex'] + 
                        row['Predicted_scores_DrugBAN'] + row['Predicted_scores_TransformerCPI']) / 4
        elif mode == 3:
            pred_sum = (row[f'label_predict_{models_mode[0]}'] + row[f'label_predict_{models_mode[1]}'] + 
                       row[f'label_predict_{models_mode[2]}'])
            avg_score = (row[f'Predicted_scores_{models_mode[0]}'] + row[f'Predicted_scores_{models_mode[1]}'] + 
                        row[f'Predicted_scores_{models_mode[2]}']) / 3
        elif mode == 2:
            pred_sum = row[f'label_predict_{models_mode[0]}'] + row[f'label_predict_{models_mode[1]}']
            avg_score = (row[f'Predicted_scores_{models_mode[0]}'] + row[f'Predicted_scores_{models_mode[1]}']) / 2
        else:
            continue

        if pred_sum == mode:
            df_trans.at[index, 'diff'] = 1
        elif pred_sum == 0:
            df_trans.at[index, 'diff'] = 0
        else:
            df_trans.at[index, 'diff'] = 2

        df_trans.at[index, 'Predicted_scores_integrated_ave'] = avg_score
        df_trans.at[index, 'label_predict_integrated_ave'] = 1 if avg_score >= 0.5 else 0

        df_trans.at[index, 'label_predict_integrated>=4'] = 1 if pred_sum >= 4 else 0
        df_trans.at[index, 'label_predict_integrated>=3'] = 1 if pred_sum >= 3 else 0
        df_trans.at[index, 'label_predict_integrated>=2'] = 1 if pred_sum >= 2 else 0
        df_trans.at[index, 'label_predict_integrated>=1'] = 1 if pred_sum >= 1 else 0

    os.makedirs(fr'{output_dir}/end_adaptivehit', exist_ok=True)
    df_trans.to_csv(fr'{output_dir}/end_adaptivehit/{file_name[:-4]}_{mode}_{models_mode[-1]}.csv', index=False)

def integ_data_creat_1(datasets, data_dir, mode):
    models = ['ConPLex', 'DeepConv-DTI', 'DrugBAN']
    error_paths = []

    for dataset in datasets:
        print(dataset)
        csv_path_trans = fr"{data_dir}/TransformerCPI/results_adaptivehit/{dataset}.csv"
        df_trans = pd.read_csv(csv_path_trans)

        if mode == 'predict':
            df_trans.columns = ['Compound_ID', 'Protein_ID', 'Predicted_scores', 'label_predict']
            df_trans.rename(columns={'Predicted_scores': 'Predicted_scores_TransformerCPI',
                                    'label_predict': 'label_predict_TransformerCPI'}, inplace=True)
        else:
            df_trans.columns = ['Compound_ID', 'Protein_ID', 'Predicted_scores', 'label_predict', 'label_original']
            df_trans.rename(columns={'Predicted_scores': 'Predicted_scores_TransformerCPI',
                                    'label_predict': 'label_predict_TransformerCPI',
                                    'label_original': 'label_origin'}, inplace=True)

        df_trans = df_trans.drop_duplicates(subset=['Compound_ID', 'Protein_ID'])
        output_dir = data_dir
        os.makedirs(output_dir, exist_ok=True)

        has_error = False
        for model in models:
            csv_path_other = fr"{data_dir}/{model}/results_adaptivehit/{dataset}.csv"
            try:
                df_other = pd.read_csv(csv_path_other)
                columns_to_select = ['Compound_ID', 'Protein_ID', 'Predicted_scores', 'label_predict']
                df_other = df_other[columns_to_select]
                df_other.rename(columns={'Predicted_scores': f'Predicted_scores_{model}',
                                        'label_predict': f'label_predict_{model}'}, inplace=True)
                df_other = df_other.drop_duplicates(subset=['Compound_ID', 'Protein_ID'])
                df_trans = pd.merge(df_trans, df_other, on=['Compound_ID', 'Protein_ID'], how='left')
            except Exception as e:
                error_paths.append(dataset)
                has_error = True
                break

        if has_error:
            continue

        df_trans.to_csv(f'{output_dir}/{dataset}.csv', index=True)

    if error_paths:
        error_df = pd.DataFrame({'Error_Paths': error_paths})
        error_df.to_csv(f'{output_dir}/notexist.csv', index=False)

def integ_data_creat_2(datasets, data_dir, mode):
    output_dir = data_dir

    error_df_path = f'{output_dir}/notexist.csv'
    error_paths_list = []
    if os.path.exists(error_df_path):
        error_df = pd.read_csv(error_df_path)
        error_paths_list = error_df['Error_Paths'].tolist()

    for dataset in datasets:
        if dataset in error_paths_list:
            continue

        df_trans = pd.read_csv(f'{output_dir}/{dataset}.csv')
        file_name = f'{dataset}.csv'

        modes = [(4, ['TransformerCPI', 'DeepConv-DTI', 'ConPLex', 'DrugBAN', 'all'])]

        for mode_val, models_mode in modes:
            ave_vote_data(df_trans, mode_val, models_mode, output_dir, file_name)

def main():
    data_dir = sys.argv[1]
    mode = sys.argv[2]

    os.makedirs(fr'{data_dir}/end_adaptivehit', exist_ok=True)

    folder_path = fr'{data_dir}/TransformerCPI/results_adaptivehit'
    csv_files = data_file_dir_csv(folder_path)

    integ_data_creat_1(csv_files, data_dir, mode)
    integ_data_creat_2(csv_files, data_dir, mode)

if __name__ == "__main__":
    main()