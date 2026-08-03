import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit.Chem.Scaffolds import MurckoScaffold
import os

# output = r'D:\.destbook\IDRB_AI\CPI\INtergratedCPI\mission_sub\差异部分重要性\bindingdb\target_select'
# csv_names = ['194-75', '488-179', '591-55', '1336-137', '212-40']

# output = r'D:\.destbook\IDRB_AI\CPI\INtergratedCPI\mission_sub\差异部分重要性\bindingdb\target_select'
# csv_names = ['212-40']

# output = r'D:\.destbook\IDRB_AI\CPI\INtergratedCPI\mission_sub\差异部分重要性\biosnap\target_select'
# csv_names = ['62-11', '103-11', '581-18']


# csv_names = ['581-18']


# output = r'D:\.destbook\IDRB_AI\CPI\INtergratedCPI\mission_sub\差异部分重要性\chembl\target_select'
# csv_names = ['1016-28']
# # csv_names = ['676-16']
# 读取 CSV 文件

def draw_plot(df, csv_name):
    # df = pd.read_csv(fr'{output}/{csv_name}.csv')  # 替换为你的 CSV 文件路径
    
    smiles_list = df['SMILES'].tolist()  # 假设 SMILES 列名为 'SMILES'

    # 生成分子指纹
    fingerprints = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            # 生成分子指纹，例如，使用 Morgan 指纹
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fingerprints.append(fp)
        else:
            fingerprints.append(None)
    print('fingerprints生成完成')

    # 去除 None 值
    fingerprints = [fp for fp in fingerprints if fp is not None]

    # 将指纹转换为 NumPy 数组
    fingerprint_array = [list(fp) for fp in fingerprints]
    print('指纹转换为 NumPy 数组完成')

    # perplexity_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
    perplexity_list = [5]
    early_exaggeration_list = [4]
    # early_exaggeration_list = [4, 8, 12, 16, 20, 24, 28, 32, 36]
    # 使用 t-SNE 降维
    for perplexity in perplexity_list:
        for early_exaggeration in early_exaggeration_list:
            tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=600, random_state=888)
            tsne_results = tsne.fit_transform(fingerprint_array)
            print('t-SNE 降维完成')

            models = ['ConPLex', 'DeepConv-DTI', 'DrugBAN', 'TransformerCPI', 'integrated_ave', 'all']
            for model in models:
                diff_labels = df[f'label_predict_{model}'].tolist()  # 假设 diff_label 列名为 'diff_label'
                print('文件已读取')

                valid_diff_labels = [label for label, fp in zip(diff_labels, fingerprints) if fp is not None]
                print(valid_diff_labels)

                # 绘制可视化图
                plt.figure(figsize=(8, 6))
                # 将 diff_label 转换为颜色
                unique_labels = list(set(diff_labels))
                print(unique_labels)
                unique_labels = list(map(str, unique_labels))
                # colors = plt.cm.get_cmap('viridis', len(unique_labels))  # 使用颜色映射

                # 定义标签对应的颜色
                custom_colors = {
                    # 'T': '#88DB29',  # 绿色
                    '200': '#FFA500',  # 橙色
                    '211': '#0000FF',  # 蓝色
                    'V': '#9617ff',  # 紫色
                    '210': '#1C1C1C',  # 橙色
                    '201': '#1C1C1C',  # 蓝色
                    
                    'T1': '#88DB29',  # 绿色
                    'T0': '#828282',  # 灰色
                    '00': '#917391',  # 紫色
                    '01': '#1C1C1C',  # 黑色
                    '11': '#1BA3E9',  # 蓝色
                    '10': '#1C1C1C',  # 黑色
                    '21': '#E32121',  # 红色
                    '20': '#EEEE00' # 黄色
                    # 可以继续添加更多标签和颜色
                }

                for i, label in enumerate(unique_labels):
                    # 获取当前标签对应的索引
                    indices = [j for j, x in enumerate(valid_diff_labels) if x == label]
                    
                    # 绘制散点
                    scatter = plt.scatter(tsne_results[indices, 0], 
                                        tsne_results[indices, 1], 
                                        s=8, alpha=0.7, 
                                        color=custom_colors[label], 
                                        # color=colors(i), 
                                        label=label)

                plt.title(f'{csv_name}')
                plt.xlabel('')
                plt.ylabel('')
                plt.legend(title='diff_label', loc='best')
                plt.grid(False)
                # 保存图形，指定 DPI 为 600
                os.makedirs(fr'{output}\target_select/plot_all_models_0_V/{csv_name}', exist_ok=True)
                plt.savefig(fr'{output}\target_select/plot_all_models_0_V/{csv_name}/{model}_{csv_name}_{perplexity}_{early_exaggeration}.png', dpi=600, bbox_inches='tight')   
                print(fr'{csv_name}_{perplexity}_{early_exaggeration}  done')

def draw_plot_core(df, csv_name):
    # df = pd.read_csv(fr'{output}/{csv_name}.csv')  # 替换为你的 CSV 文件路径
    # smiles_list = df['SMILES'].tolist()  # 假设 SMILES 列名为 'SMILES'
    # diff_labels = df['diff_label'].tolist()  # 假设 diff_label 列名为 'diff_label'
    df['Compound_ID'] = df['Compound_ID'].astype(str)
    smiles_list = df['Compound_ID'].tolist()  # 假设 SMILES 列名为 'SMILES'
    diff_labels = df['diff_label'].tolist()  # 假设 diff_label 列名为 'diff_label'
    print('文件已读取')

    # 生成分子指纹
    fingerprints = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            # 生成分子指纹，例如，使用 Morgan 指纹
            core1 = MurckoScaffold.GetScaffoldForMol(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fingerprints.append(fp)
        else:
            fingerprints.append(None)
    print('fingerprints生成完成')

    # 去除 None 值
    fingerprints = [fp for fp in fingerprints if fp is not None]
    valid_diff_labels = [label for label, fp in zip(diff_labels, fingerprints) if fp is not None]
    print(valid_diff_labels)

    # 将指纹转换为 NumPy 数组
    fingerprint_array = [list(fp) for fp in fingerprints]
    print('指纹转换为 NumPy 数组完成')

    # perplexity_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
    perplexity_list = [5]
    early_exaggeration_list = [4, 8, 12, 16, 20, 24, 28, 32, 36]

    # 使用 t-SNE 降维
    for perplexity in perplexity_list:
        for early_exaggeration in early_exaggeration_list:
            tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=600, random_state=888)
            tsne_results = tsne.fit_transform(fingerprint_array)
            print('t-SNE 降维完成')

            # 绘制可视化图
            plt.figure(figsize=(8, 6))
            # 将 diff_label 转换为颜色
            unique_labels = list(set(diff_labels))
            print(unique_labels)
            # colors = plt.cm.get_cmap('viridis', len(unique_labels))  # 使用颜色映射

            # 定义标签对应的颜色
            custom_colors = {
                'T': '#88DB29',  # 蓝色
                '0': '#917391',  # 橙色
                '1': '#1BA3E9',  # 绿色
                '2': '#E32121',  # 红色
                'T': '#88DB29',  # 蓝色
                0 : '#917391',  # 橙色
                1: '#1BA3E9',  # 绿色
                2: '#E32121',  # 红色
                # 可以继续添加更多标签和颜色
            }

            for i, label in enumerate(unique_labels):
                # 获取当前标签对应的索引
                indices = [j for j, x in enumerate(valid_diff_labels) if x == label]
                
                # 绘制散点
                scatter = plt.scatter(tsne_results[indices, 0], 
                                    tsne_results[indices, 1], 
                                    s=8, alpha=0.7, 
                                    color=custom_colors[label], 
                                    # color=colors(i), 
                                    label=label)

            plt.title(f'{csv_name}')
            plt.xlabel('')
            plt.ylabel('')
            plt.legend(title='diff_label', loc='best')
            plt.grid(False)
            # 保存图形，指定 DPI 为 600
            os.makedirs(fr'{output}\target_select/plot_all/{csv_name}', exist_ok=True)
            plt.savefig(fr'{output}\target_select/plot_all/{csv_name}/{csv_name}_{perplexity}_{early_exaggeration}.png', dpi=600, bbox_inches='tight')   
            print(fr'{csv_name}_{perplexity}_{early_exaggeration}  done')



def draw_list():
    df_list = pd.read_csv(fr'{output}/plot_draw_list_0.csv')  # 替换为你的 CSV 文件路径
    df_list = df_list[['Target Sequence','name_2']]
    for index, row in df_list.iterrows():
        sequence = row['Target Sequence']
        name = row['name_2']
        print(name)
        df_all_data = pd.read_csv(fr'{output}/chembl_all_preprocess.csv') 
        df_filter = df_all_data[df_all_data['Target Sequence'] == sequence]
        print(df_filter[:2])
        draw_plot(df_filter, name)

# # "D:\.destbook\IDRB_AI\CPI\INtergratedCPI\mission_sub\差异部分重要性\chembl"
# output = r"D:\.destbook\IDRB_AI\CPI\INtergratedCPI\mission_sub\差异部分重要性\chembl\analyse\cluster_scaffold_analyse"
# csv_name = 'disagreement_consensus_scaffolds_test'
# df = pd.read_csv(fr'{output}\disagreement_consensus_scaffolds_test.csv')
# print(df[:3]
# draw_plot_core(df, csv_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot Morgan-fingerprint t-SNE "chemical space" for a set of molecules '
                    '(reads {output}/plot_draw_list_0.csv and {output}/chembl_all_preprocess.csv)')
    parser.add_argument('--output', type=str, required=True,
                        help='Directory containing plot_draw_list_0.csv/chembl_all_preprocess.csv, '
                             'and where plots are written')
    args = parser.parse_args()
    output = args.output
    draw_list()