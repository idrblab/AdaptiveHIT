import os
import sys
import pandas as pd

# The 4 base models AdaptiveHIT ships. To replace or add a base model, pass
# your own list as trailing argv to main() (anchor model first) -- see the
# "Adding or Replacing a Base Model" section in README.md.
DEFAULT_BASE_MODELS = ['TransformerCPI', 'DeepConv-DTI', 'ConPLex', 'DrugBAN']

def data_file_dir_csv(folder_path):
    csv_files = [file[:-4] for file in os.listdir(folder_path) if file.endswith('.csv')]
    return csv_files

def ave_vote_data(df_trans, mode, models_mode, output_dir, file_name):
    names = models_mode[:mode]
    for index, row in df_trans.iterrows():
        if len(names) != mode:
            continue
        pred_sum = sum(row[f'label_predict_{m}'] for m in names)
        avg_score = sum(row[f'Predicted_scores_{m}'] for m in names) / mode

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

    os.makedirs(fr'{output_dir}/end_meta', exist_ok=True)
    df_trans.to_csv(fr'{output_dir}/end_meta/{file_name[:-4]}_{mode}_{models_mode[-1]}.csv', index=False)

def integ_data_creat_1(datasets, data_dir, mode, base_models=DEFAULT_BASE_MODELS):
    anchor, models = base_models[0], base_models[1:]
    error_paths = []

    for dataset in datasets:
        print(dataset)
        csv_path_trans = fr"{data_dir}/{anchor}/results_meta/{dataset}.csv"
        df_trans = pd.read_csv(csv_path_trans)

        if mode == 'predict':
            df_trans.columns = ['Compound_ID', 'Protein_ID', 'Predicted_scores', 'label_predict']
            df_trans.rename(columns={'Predicted_scores': f'Predicted_scores_{anchor}',
                                    'label_predict': f'label_predict_{anchor}'}, inplace=True)
        else:
            df_trans.columns = ['Compound_ID', 'Protein_ID', 'Predicted_scores', 'label_predict', 'label_original']
            df_trans.rename(columns={'Predicted_scores': f'Predicted_scores_{anchor}',
                                    'label_predict': f'label_predict_{anchor}',
                                    'label_original': 'label_origin'}, inplace=True)

        df_trans = df_trans.drop_duplicates(subset=['Compound_ID', 'Protein_ID'])
        output_dir = data_dir
        os.makedirs(output_dir, exist_ok=True)

        has_error = False
        for model in models:
            csv_path_other = fr"{data_dir}/{model}/results_meta/{dataset}.csv"
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

def integ_data_creat_2(datasets, data_dir, mode, base_models=DEFAULT_BASE_MODELS):
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

        modes = [(len(base_models), base_models + ['all'])]

        for mode_val, models_mode in modes:
            ave_vote_data(df_trans, mode_val, models_mode, output_dir, file_name)

def main():
    data_dir = sys.argv[1]
    mode = sys.argv[2]
    base_models = sys.argv[3:] if len(sys.argv) > 3 else DEFAULT_BASE_MODELS

    os.makedirs(fr'{data_dir}/end_meta', exist_ok=True)

    folder_path = fr'{data_dir}/{base_models[0]}/results_meta'
    csv_files = data_file_dir_csv(folder_path)

    integ_data_creat_1(csv_files, data_dir, mode, base_models)
    integ_data_creat_2(csv_files, data_dir, mode, base_models)

if __name__ == "__main__":
    main()