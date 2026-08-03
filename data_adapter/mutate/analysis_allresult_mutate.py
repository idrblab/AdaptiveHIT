import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
import sys
import numpy as np

def data_file_dir_csv(folder_path):
    # 获取文件夹中的所有 CSV 文件，排除子文件夹中的文件
    csv_files = [os.path.splitext(file)[0] for file in os.listdir(folder_path) 
                 if file.endswith('.csv')]
    return csv_files

def metrics(correct_labels, predicted_labels):
    ACC = accuracy_score(correct_labels, predicted_labels)
    CM = confusion_matrix(correct_labels, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    F1 = 2 * Pre * Rec / (Pre + Rec) if (Pre + Rec) > 0 else 0
    False_Positive = FP/(FP + TN) if (FP + TN) > 0 else 0
    MCC = matthews_corrcoef(correct_labels, predicted_labels)
    return ACC, Rec, Pre, F1, MCC, False_Positive

def data_pair_label_merge_pdb(result_dir, label_dir, file_name):
    # pair原标签对齐
    result_df = pd.read_csv(f'{result_dir}/end_contact/{file_name}.csv')
    label_df = pd.read_csv(f'{label_dir}')
    label_df.columns=['index','Protein_ID','Compound_ID','label', 'mutated_position', 'PDB_ID','Ligand_Name','siteslabel']
    df_end = pd.merge(result_df, label_df, on=['Protein_ID','Compound_ID'], how='left')
    df_end.to_csv(f'{result_dir}/end_pair/{file_name}_pair.csv', index=False)
    return df_end

def normalize_values(values, method='minmax'):
    """
    归一化方法
    method: 'minmax' - Min-Max归一化
            'log' - 对数变换后Min-Max归一化
            'robust' - 基于中位数和四分位距的鲁棒归一化
            'zscore' - Z-score标准化后使用sigmoid映射到[0,1]
    """
    values = np.array(values)
    
    if method == 'minmax':
        # 原始的Min-Max归一化
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(values)
            
    elif method == 'log':
        # 对数变换后Min-Max归一化
        log_values = np.log1p(values)  # log(1+x)
        min_val = np.min(log_values)
        max_val = np.max(log_values)
        if max_val > min_val:
            normalized = (log_values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(log_values)
            
    elif method == 'robust':
        # 基于中位数和四分位距的鲁棒归一化
        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if iqr > 0:
            # 使用中位数和IQR进行标准化
            normalized = (values - median) / iqr
            # 使用tanh将值映射到[-1,1]区间，再转换到[0,1]
            normalized = (np.tanh(normalized) + 1) / 2
        else:
            normalized = np.zeros_like(values)
    
    elif method == 'zscore':
        # Z-score标准化后使用sigmoid映射到[0,1]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val > 0:
            # 计算Z-score
            z_scores = (values - mean_val) / std_val
            # 使用sigmoid函数将Z-score映射到[0,1]区间
            # sigmoid: 1 / (1 + e^(-x))
            normalized = 1 / (1 + np.exp(-z_scores))
        else:
            # 如果标准差为0，所有值相同，返回0.5
            normalized = np.full_like(values, 0.5)
            
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.tolist()

def get_threshold_and_labels(normalized_values, ori_label, threshold_method='percentile', cutoff=0.05):
    """
    获取阈值并生成标签
    threshold_method: 'percentile' - 使用前cutoff百分比的样本标记为1
                      'count' - 使用ori_label中1的数量来标记前N个样本为1
    cutoff: 当method='percentile'时，表示百分比（默认0.05即前5%）
    """
    if threshold_method == 'percentile':
        # 原来的方法：计算前cutoff%的阈值
        sorted_values = sorted(normalized_values)
        threshold_index = int(len(sorted_values) * (1 - cutoff))
        threshold_index = max(0, min(threshold_index, len(sorted_values) - 1))
        threshold_value = sorted_values[threshold_index]
        labels = [1 if x >= threshold_value else 0 for x in normalized_values]
        
    elif threshold_method == 'count':
        # 使用ori_label中1的数量来确定标记多少个样本
        n_positive = sum(ori_label)
        if n_positive == 0:
            # 如果没有正样本，全部标记为0
            labels = [0] * len(normalized_values)
        else:
            # 获取normalized_values中值最大的前n_positive个样本的索引
            n_positive = min(n_positive, len(normalized_values))  # 确保不超过总样本数
            sorted_indices = np.argsort(normalized_values)[::-1]  # 降序排列
            labels = [0] * len(normalized_values)
            for i in range(n_positive):
                labels[sorted_indices[i]] = 1
                
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    return labels

def data_process(df_end, output_dir, cutoff, models, 
                 normalization_method='minmax', 
                 threshold_method='percentile'):
    """
    处理突变数据
    normalization_method: 'minmax', 'log', 'robust'
    threshold_method: 'percentile', 'count'
    """
    # 加载 CSV 数据
    df = df_end

    mutated_positions = df['mutated_position'].drop_duplicates().tolist()
    value_to_remove = 0
    filtered_values = list(filter(lambda x: x != value_to_remove, mutated_positions))

    df_end_data = pd.DataFrame()
    df_end_data['mutated_position'] = filtered_values
    filtered_df_0_label = df[df['mutated_position'] == 0]
    try:
        ori_label = list(filtered_df_0_label.loc[0, 'siteslabel'])
    except KeyError as e:
        print('filtered_df_0_label.empty')
        return  
    
    df_end_data[f'ori_label'] = ori_label
    ori_label = [int(label_1) for label_1 in ori_label]

    for model in models:
        average_differencelist = []
        filtered_df_0 = df[df['mutated_position'] == 0]
        adjustment_value = filtered_df_0.loc[0, f'Predicted_scores_{model}']
        PDB_ID = filtered_df_0.loc[0, 'PDB_ID']
        Ligand_Name = filtered_df_0.loc[0, 'Ligand_Name']
        
        for mutated_position in filtered_values:
            filtered_df = df[df['mutated_position'] == mutated_position]
            values_list = filtered_df[f'Predicted_scores_{model}'].tolist()
            # 计算绝对差值
            absolute_differences = [abs(value - adjustment_value) for value in values_list]
            # 计算平均值
            average_difference = sum(absolute_differences) / len(absolute_differences) if absolute_differences else 0
            average_differencelist.append(average_difference)

        # 使用指定的归一化方法
        normalized_averages = normalize_values(average_differencelist, method=normalization_method)
        
        # 使用指定的阈值方法生成标签
        label = get_threshold_and_labels(normalized_averages, ori_label, threshold_method, cutoff)
        
        # 保存结果
        df_end_data[f'mutated_value_{model}'] = normalized_averages
        df_end_data[f'ori_value_{model}'] = average_differencelist
        df_end_data[f'mutated_label_{model}'] = label

        # 计算评估指标
        ACC, Rec, Pre, F1, MCC, FP = metrics(ori_label, label)
        df_evalu = pd.DataFrame([[PDB_ID, Ligand_Name, model, ACC, Rec, Pre, F1, MCC, FP]],
                               columns=['PDB_ID', 'Ligand_Name', 'model', 'ACC', 'Rec', 'Pre', 'F1', 'MCC', 'FP'])
        df_evalu.to_csv(fr'{output_dir}/data_metrics/integrated_evaluation.csv', index=False, mode='a', 
                       header=not os.path.exists(fr'{output_dir}/data_metrics/integrated_evaluation.csv'))
        
    df_end_data.to_csv(fr'{output_dir}/data_analyse/{PDB_ID}_{Ligand_Name}.csv', index=False)

def data_concat(folder_path, output_path):
    # 创建一个空的列表来存储 DataFrame
    dataframes = []
    # 遍历文件夹中的所有 CSV 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            # 将 DataFrame 添加到列表中
            dataframes.append(df)

    # 拼接所有 DataFrame
    # ignore_index=True 确保索引重新排列
    final_df = pd.concat(dataframes, ignore_index=True)
    print(len(final_df))
    print(final_df.shape)
    # 保存为新的 CSV 文件
    final_df.to_csv(fr'{output_path}/combined_DUDE_result.csv', index=False)  # 替换为您想要的输出文件名

    print("所有 CSV 文件已成功拼接并保存为 combined_output.csv")

def concatenate_csv_files(input_folder, prefix, output_file):
    # 获取指定文件夹中所有以 prefix 开头的 CSV 文件
    csv_files = [f for f in os.listdir(input_folder) if f.startswith(prefix) and f.endswith('.csv')]
    
    # 初始化一个空的列表，用于存储 DataFrame
    dataframes = []
    
    # 遍历所有找到的 CSV 文件并读取
    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    # 使用 pd.concat 拼接所有 DataFrame
    if dataframes:
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        
        # 保存拼接后的 DataFrame 到指定路径
        concatenated_df.to_csv(fr'{output_file}/{csv_files[0]}.csv', index=False)
        print(f"拼接后的 DataFrame 已保存到: {output_file}")
    else:
        print("没有找到符合条件的 CSV 文件。")

# 主程序
label_csv_dir = sys.argv[1]
file_name = sys.argv[2]
cutoff = 0.05  # 当threshold_method='percentile'时使用，表示前5%
normalization_method = sys.argv[3]
threshold_method = sys.argv[4]
# 可选参数（可以通过命令行参数传入，这里先硬编码）
# normalization_method = 'zscore'  # 可选: 'minmax', 'log', 'robust', 'zscore'
# threshold_method = 'count'    # 可选: 'percentile', 'count'

models = ['ConPLex', 'DeepConv-DTI', 'TransformerCPI', 'DrugBAN', 'average', 
          'weighted_logistic_balanced', 
          'meta_full_esm2_xmol_prob_attention-16-0.0005-none-optimizer_new-normal-none-0.5-auto-truexmol']

result_dir = fr'{label_csv_dir}/{file_name}_{normalization_method}_{threshold_method}'
os.makedirs(f'{result_dir}/end_pair', exist_ok=True) 
os.makedirs(f'{result_dir}/data_analyse', exist_ok=True) 
os.makedirs(f'{result_dir}/data_metrics', exist_ok=True) 
os.makedirs(f'{result_dir}/end_contact', exist_ok=True) 

# 创建评估结果文件（带参数标识）
eval_filename = f"integrated_evaluation_{normalization_method}_{threshold_method}.csv"
df_evalu = pd.DataFrame(columns=['PDB_ID', 'Ligand_Name', 'model', 'ACC', 'Rec', 'Pre', 'F1', 'MCC', 'FP'])
df_evalu.to_csv(fr'{result_dir}/data_metrics/{eval_filename}', index=False)

df_end = pd.read_csv(f'{label_csv_dir}/{file_name}.csv')
data_process(df_end, result_dir, cutoff, models, 
            normalization_method=normalization_method, 
            threshold_method=threshold_method)