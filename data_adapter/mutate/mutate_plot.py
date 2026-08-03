import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def create_mutation_plots(PDB_list, input_folder, output_folder, models):
    """
    创建突变位置柱状图
    
    Args:
        PDB_list: PDB文件列表
        input_folder: 输入文件所在文件夹
        output_folder: 输出文件夹
        models: 模型列表
    """
    for PDB in PDB_list:
        print(f"处理: {PDB}")
        
        # 读取数据
        df = pd.read_csv(os.path.join(input_folder, f"{PDB}.csv"))
        output_path = os.path.join(output_folder, 'plot_select_opt', PDB)
        os.makedirs(output_path, exist_ok=True)
        
        # 为每个模型创建图表
        for model in models:
            plot_model(df, model, output_path, PDB)

def plot_model(df, model, output_path, pdb_id):
    """
    绘制单个模型的柱状图
    
    Args:
        df: 数据框
        model: 模型名称
        output_path: 输出路径
        pdb_id: PDB ID
    """
    value_col = f'mutated_value_{model}'
    label_col = f'mutated_label_{model}'
    
    # 检查必要列是否存在
    if value_col not in df.columns or label_col not in df.columns:
        print(f"  跳过模型 {model}：缺少必要列")
        return
    
    # 计算阈值
    threshold = calculate_threshold(df, value_col, label_col)
    
    # 根据真实标签和预测标签的组合设置颜色
    # 规则: 
    # - 真实0, 预测0 (TN): 灰色 '#DCDCDC'
    # - 真实0, 预测1 (FP): 暗黄色 '#FDCF9E'
    # - 真实1, 预测0 (FN): 蓝色 '#5C6A85'
    # - 真实1, 预测1 (TP): 红色 '#EF786C'
    colors = []
    site_colors = []  # 用于竖线的颜色
    for idx in range(len(df)):
        true_label = df['ori_label'].iloc[idx]
        pred_label = df[label_col].iloc[idx]
        
        if true_label == 0 and pred_label == 0:  # TN
            bar_color = '#DCDCDC'
            line_color = '#DCDCDC'
        elif true_label == 0 and pred_label == 1:  # FP
            bar_color = '#FDCF9E'
            line_color = '#FDCF9E'
        elif true_label == 1 and pred_label == 0:  # FN
            bar_color = '#5C6A85'
            line_color = '#5C6A85'
        else:  # true_label == 1 and pred_label == 1  # TP
            bar_color = '#EF786C'
            line_color = '#EF786C'
        
        colors.append(bar_color)
        site_colors.append(line_color)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制柱状图
    ax.bar(df['mutated_position'], df[value_col], color=colors, width=1, zorder=2)
    




    # 获取当前Y轴的范围
    y_min, y_max = ax.get_ylim()
    
    # 计算竖线的长度（Y轴范围的3%）
    line_length = (y_max - y_min) * 0.03
    
    # 为所有位点添加竖线，颜色与柱状图颜色一致
    for idx, (site, line_color) in enumerate(zip(df['mutated_position'], site_colors)):
        ax.plot([site, site], [y_min - line_length, y_min], 
               color=line_color, linewidth=2.2, clip_on=False, zorder=3)


    # 添加阈值线
    if threshold is not None:
        ax.axhline(y=threshold, color='#EF786C', linestyle='--', 
                   linewidth=2, label=f'阈值={threshold:.6f}', zorder=4)
        ax.axhline(y=0, color='black', linestyle='-', 
                linewidth=0.6, zorder=5)

    # 设置图表属性
    ax.set_title(f'{model} - {pdb_id}', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'{output_path}/{model}.png', dpi=600)
    plt.close()
    
    print(f"  已保存: {model}.png (阈值={threshold if threshold is not None else 'N/A'})")

def calculate_threshold(df, value_col, label_col):
    """
    计算阈值
    
    Args:
        df: 数据框
        value_col: 值列名
        label_col: 标签列名
    
    Returns:
        阈值或None
    """
    positive_values = df[df[label_col] == 1][value_col]
    return positive_values.min() if not positive_values.empty else None

import sys
input_folder = sys.argv[1]

# 配置参数
input_folder = fr'{input_folder}/data_analyse'
output_folder = fr'{input_folder}/plot2'

models = ['ConPLex', 'DeepConv-DTI', 'TransformerCPI', 'DrugBAN', 'average', 
          'weighted_logistic_balanced', 
          'meta_full_esm2_xmol_prob_attention-16-0.0005-none-optimizer_new-normal-none-0.5-auto-truexmol']
PDB_list = ['4zts_4RK']

# models = ['ConPLex', 'DeepConv-DTI', 'TransformerCPI', 'DrugBAN', 'integrated_ave']
# PDB_list = ['9IZD_1000163']
# 执行绘图
create_mutation_plots(PDB_list, input_folder, output_folder, models)

print("所有图表生成完成！")