# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/17 8:54
@author: LiFan Chen
@Filename: main.py
@Software: PyCharm
"""
import torch
import numpy as np
import random
import os
import time
from model import *
import timeit
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import gc


# 修复的批处理函数 - 返回模型需要的所有参数
def optimized_stack_fn(batch):
    """修复的批处理函数，确保所有张量在同一个设备上"""
    compounds, adjacencies, proteins, interactions = zip(*batch)
    
    # 计算每个样本的实际长度（用于mask）
    atom_num = [c.shape[0] for c in compounds]
    protein_num = [p.shape[0] for p in proteins]
    
    # 对化合物和蛋白质使用pad_sequence
    compounds_padded = torch.nn.utils.rnn.pad_sequence(
        compounds, batch_first=True, padding_value=0.0
    )
    proteins_padded = torch.nn.utils.rnn.pad_sequence(
        proteins, batch_first=True, padding_value=0.0
    )
    
    # 对邻接矩阵进行特殊处理 - 找到最大的维度
    max_dim = max(adj.shape[0] for adj in adjacencies)
    adjacencies_padded = []
    
    for adj in adjacencies:
        current_dim = adj.shape[0]
        if current_dim < max_dim:
            # 填充到最大维度
            padding_size = max_dim - current_dim
            padded_adj = torch.nn.functional.pad(
                adj, 
                (0, padding_size, 0, padding_size),
                mode='constant', 
                value=0
            )
            adjacencies_padded.append(padded_adj)
        else:
            adjacencies_padded.append(adj)
    
    adjacencies_padded = torch.stack(adjacencies_padded)
    interactions_tensor = torch.cat(interactions)
    
    return compounds_padded, adjacencies_padded, proteins_padded, interactions_tensor, atom_num, protein_num

# 优化的DTI数据集类
class OptimizedDTIDataset(Dataset):
    def __init__(self, device, data_dir, label_file_path=None, datatype="train"):
        super(OptimizedDTIDataset, self).__init__()
        
        self.device = device
        self.data_dir = data_dir
        self.datatype = datatype
        
        # 读取标签文件
        self.df_interactions = pd.read_csv(label_file_path)
        
        # 一次性加载所有特征字典到内存
        self.compound_dict = self._load_feature_dict('compounds.npy')
        self.adjacency_dict = self._load_feature_dict('adjacencies.npy') 
        self.protein_dict = self._load_feature_dict('proteins.npy')
        
        self.label_list = self.df_interactions.iloc[:, 2].values
        
        # 预加载所有样本的ID
        self.compound_ids = self.df_interactions.iloc[:, 0].values
        self.protein_ids = self.df_interactions.iloc[:, 1].values

    def _load_feature_dict(self, filename):
        """加载特征字典到内存"""
        filepath = os.path.join(self.data_dir, filename)
        return np.load(filepath, allow_pickle=True).item()

    def __len__(self):
        return len(self.df_interactions)

    def __getitem__(self, index):
        # 直接从内存字典获取，避免文件I/O
        compound_id = self.compound_ids[index]
        protein_id = self.protein_ids[index]
        
        # 直接转换为tensor，避免不必要的复制
        compound_tensor = torch.from_numpy(self.compound_dict[compound_id]).float()
        adjacency_tensor = torch.from_numpy(self.adjacency_dict[compound_id]).float()
        protein_tensor = torch.from_numpy(self.protein_dict[protein_id]).float()
        label_tensor = torch.FloatTensor([self.label_list[index]])
        
        return compound_tensor, adjacency_tensor, protein_tensor, label_tensor


# 内存高效的训练器 - 使用model.py中的Trainer
class CompatibleTrainer:
    def __init__(self, model, lr, weight_decay, batch_size):
        # 使用model.py中的Trainer
        self.trainer = Trainer(model, lr, weight_decay, batch_size)
        self.device = model.device
    
    def train_batch(self, batch_data):
        """使用model.py中的训练逻辑，确保设备一致"""
        # 首先将所有数据移动到GPU
        compounds, adjacencies, proteins, interactions, atom_num, protein_num = batch_data
        
        compounds = compounds.to(self.device, non_blocking=True)
        adjacencies = adjacencies.to(self.device, non_blocking=True)
        proteins = proteins.to(self.device, non_blocking=True)
        interactions = interactions.to(self.device, non_blocking=True)
        
        # model.py中的Trainer期望的是dataset格式的数据
        # 我们需要将batch数据转换为model.py期望的格式
        dataset_batch = []
        for i in range(len(compounds)):
            compound = compounds[i]
            adjacency = adjacencies[i] 
            protein = proteins[i]
            interaction = interactions[i]
            dataset_batch.append((compound, adjacency, protein, interaction))
        
        loss = self.trainer.train(dataset_batch, self.device)
        return loss


class CompatibleTester:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        # 使用model.py中的Tester
        self.tester = Tester(model)
    
    def test_with_dataloader(self, dataset, batch_size, device, collate_fn):
        """逐个batch处理验证/测试数据，避免OOM"""
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=0,
            shuffle=False  # 验证/测试不需要shuffle
        )
        
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            total_batches = len(dataloader)
            for batch_idx, batch_data in enumerate(dataloader):
                # 打印验证/测试进度
                print(f'Processing batch {batch_idx+1}/{total_batches}', end='\r')
                
                compounds, adjacencies, proteins, interactions, atom_num, protein_num = batch_data
                
                # 将数据移动到GPU
                compounds = compounds.to(device, non_blocking=True)
                adjacencies = adjacencies.to(device, non_blocking=True)
                proteins = proteins.to(device, non_blocking=True)
                interactions = interactions.to(device, non_blocking=True)
                
                # 使用model.py中的pack函数处理数据
                # 将batch数据转换为pack函数期望的格式
                atoms_list = [compounds[i] for i in range(len(compounds))]
                adjs_list = [adjacencies[i] for i in range(len(adjacencies))]
                proteins_list = [proteins[i] for i in range(len(proteins))]
                labels_list = [interactions[i] for i in range(len(interactions))]
                
                data_pack = pack(atoms_list, adjs_list, proteins_list, labels_list, device)
                
                # 使用model.py中的测试逻辑
                correct_labels, predicted_labels, predicted_scores = self.model(data_pack, train=False)
                
                # 收集结果
                all_predictions.extend(predicted_scores)
                all_labels.extend(correct_labels)
                
                # 立即清理GPU显存
                del compounds, adjacencies, proteins, interactions, data_pack
                del atoms_list, adjs_list, proteins_list, labels_list
                torch.cuda.empty_cache()
                
                # 每10个batch额外清理一次
                if batch_idx % 10 == 0:
                    gc.collect()
            
            print()  # 换行
        
        # 计算指标
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        predicted_labels = (all_predictions > 0.5).astype(int)
        
        AUC = roc_auc_score(all_labels, all_predictions)
        PRC = average_precision_score(all_labels, all_predictions)
        ACC = accuracy_score(all_labels, predicted_labels)
        Rec = recall_score(all_labels, predicted_labels)
        Pre = precision_score(all_labels, predicted_labels)
        F1 = f1_score(all_labels, predicted_labels)
        MCC = matthews_corrcoef(all_labels, predicted_labels)
        
        return AUC, PRC, ACC, Rec, Pre, F1, MCC
    
    def save_model(self, model, path):
        """保存模型"""
        torch.save(model.state_dict(), path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""
    This Python script is used to train, validate, test deep learning model for prediction of drug-target interaction                                 
    """)
    # datasets_params
    parser.add_argument("dataset_train_dir", help="dir of dataset_train [SMILES, sqeuence, label]")
    parser.add_argument("dataset_dev_dir", help="dir of dataset_dev [SMILES, sqeuence, label]")
    # training_params
    parser.add_argument("--learning-rate", '-r', help="Learning late for training", default=1e-4, type=float)
    parser.add_argument("--n-epoch", '-e', help="The number of epochs for training or validation", type=int, default=15)
    parser.add_argument("--dropout", "-D", help="Dropout ratio", default=0.2, type=float)
    parser.add_argument("--batch-size", "-b", help="Batch size", default=32, type=int)
    # output_params
    parser.add_argument("--output-model", "-m", help="save model", type=str)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)
    args = parser.parse_args()

    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed dev/test data directories."""
    dir_input_train = args.dataset_train_dir
    dir_input_dev = args.dataset_dev_dir
    dir_input_test = f'{args.dataset_dev_dir[:-5]}/test/'

    """ create model ,trainer and tester """
    dropout = args.dropout
    batch_size = args.batch_size
    lr = args.learning_rate
    n_epoch = args.n_epoch
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    kernel_size = 7

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.to(device)
    trainer = CompatibleTrainer(model, lr, weight_decay, batch_size)
    tester = CompatibleTester(model)

    """Output files."""
    output_file = args.output
    file_model = args.output_model

    """Start training."""
    print('Training...')

    # 预加载所有数据集
    print("Loading datasets...")
    train_dataset = OptimizedDTIDataset(device, dir_input_train, dir_input_train + 'dti.csv', "train")
    valid_dataset = OptimizedDTIDataset(device, dir_input_dev, dir_input_dev + 'dti.csv', "dev")
    test_dataset = OptimizedDTIDataset(device, dir_input_test, dir_input_test + 'dti.csv', "test")

    max_AUC_dev = 0
    result_dic = {
        "learning_rate":[], "batch_size":[], "dropout":[], "n_epoch":[], 
        "AUC":[], "AUPR": [], "ACC":[], "Rec":[], "Pre":[], "F1":[], "MCC":[],
        "AUC_test":[], "AUPR_test":[], "ACC_test":[], "Rec_test":[], 
        "Pre_test":[], "F1_test":[], "MCC_test":[]
    }
    best_model_AUC = 0
    last_improve = 0

    start = timeit.default_timer()

    for epoch_num in range(1, n_epoch+1):
        epoch_start = time.time()
        print(f'Epoch {epoch_num}:')
        
        # 学习率衰减
        if epoch_num % decay_interval == 0:
            trainer.trainer.optimizer.param_groups[0]['lr'] *= lr_decay
            print(f"Learning rate decayed to: {trainer.trainer.optimizer.param_groups[0]['lr']}")

        # 创建训练DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=optimized_stack_fn,
            pin_memory=True,
            shuffle=True
        )
        
        # 训练阶段
        model.train()
        total_loss = 0
        batch_count = 0
        
        total_batches = len(train_loader)
        for batch_idx, batch_data in enumerate(train_loader):
            # 训练并获取损失
            batch_loss = trainer.train_batch(batch_data)
            total_loss += batch_loss
            batch_count += 1
            
            # 打印进度信息
            print(f'Epoch {epoch_num}: Batch {batch_idx+1}/{total_batches} | Loss: {batch_loss:.4f}', end='\r')
            
            # 及时清理显存
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if batch_count % 10 == 0:
                current_memory = torch.cuda.memory_allocated()/1024**3 if torch.cuda.is_available() else 0
                print(f'\nBatch {batch_count}/{total_batches}, Loss: {batch_loss:.4f}, GPU Mem: {current_memory:.2f}GB')

        # 完成一个epoch后换行
        print()

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch_num} Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch_num}: No batches processed")
            continue
        
        # 验证阶段
        print("Validating...")

        # 验证前彻底清理显存
        torch.cuda.empty_cache()
        gc.collect()

        AUC, PRC, ACC, Rec, Pre, F1, MCC = tester.test_with_dataloader(
            valid_dataset, batch_size, device, optimized_stack_fn
        )
        epoch_time = time.time() - epoch_start
        print('Time/epoch_time', epoch_time, epoch_num)

        # 测试阶段
        print("Testing...")
        epoch_start_predict = time.time()
        # 测试前彻底清理显存
        torch.cuda.empty_cache()
        gc.collect()

        AUC_test, PRC_test, ACC_test, Rec_test, Pre_test, F1_test, MCC_test = tester.test_with_dataloader(
            test_dataset, batch_size, device, optimized_stack_fn
        )

        print("\tDev - AUC: %0.3f, AUPR: %0.3f, ACC: %0.3f" % (AUC, PRC, ACC))
        print("\tTest - AUC: %0.3f, AUPR: %0.3f, ACC: %0.3f" % (AUC_test, PRC_test, ACC_test))
        epoch_time_predict = time.time() - epoch_start_predict
        print('Time/predict_time', epoch_time_predict, epoch_num)
        # 保存结果
        result_dic["learning_rate"].append(lr)
        result_dic["batch_size"].append(batch_size)
        result_dic["dropout"].append(dropout)
        result_dic["n_epoch"].append(epoch_num)
        result_dic["AUC"].append(AUC)
        result_dic["AUPR"].append(PRC)
        result_dic["ACC"].append(ACC)
        result_dic["Rec"].append(Rec)
        result_dic["Pre"].append(Pre)
        result_dic["F1"].append(F1)
        result_dic["MCC"].append(MCC)
        result_dic["AUC_test"].append(AUC_test)
        result_dic["AUPR_test"].append(PRC_test)
        result_dic["ACC_test"].append(ACC_test)
        result_dic["Rec_test"].append(Rec_test)
        result_dic["Pre_test"].append(Pre_test)
        result_dic["F1_test"].append(F1_test)
        result_dic["MCC_test"].append(MCC_test)

        


        # 保存模型
        model_suffix = file_model[:-6] if file_model.endswith('.model') else file_model
        tester.save_model(model, f'{model_suffix}_epo{epoch_num}.model')
        if AUC >= best_model_AUC:
            last_improve = epoch_num
            best_model_AUC = AUC
            tester.save_model(model, f'{model_suffix}_best.model')
            tester.save_model(model, file_model)  # exact --output-model path, expected by pipeline/predict/*.sh and predict_trans.py's model_dir arg
            print(f"Best model saved! AUC: {AUC:.3f}")
        
        # end_epoch = timeit.default_timer()
        # time_elapsed = end_epoch - start
        print(f'Epoch {epoch_num} completed in {epoch_time:.2f}s, last_improve: {last_improve}')
        
        # 早停
        if epoch_num - last_improve >= 10:
            print(f'Early stopping at epoch: {epoch_num}')
            break

        # 清理验证和测试产生的显存
        torch.cuda.empty_cache()
        gc.collect()

    # 保存最终结果
    print(f"Results saved to {output_file}, Best AUC: {best_model_AUC:.3f}")
    result_df = pd.DataFrame(result_dic)
    result_df.to_csv(output_file, index=False, encoding='utf8')