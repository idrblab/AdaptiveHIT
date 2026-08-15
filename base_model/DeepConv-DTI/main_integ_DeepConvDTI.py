# import numpy and pandas
import numpy as np
import pandas as pd
import random
import h5py
import time
import os  
import tensorflow as tf
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Embedding, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2,l1
from keras.preprocessing import sequence
from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef

# 配置GPU显存使用
# def setup_gpu_memory(gpu_memory_fraction=0.8):
#     """配置GPU显存使用，避免内存浪费"""
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True  # 按需增长显存
#     config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
#     session = InteractiveSession(config=config)
#     return session

seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [seq_dic[aa] for aa in seq]

def optimized_parse_data(dti_dir, drug_dir, protein_dir, with_label=True,
                        prot_len=2500, prot_vec="Convolution",
                        drug_vec="Convolution", drug_len=2048):
    """
    优化版数据解析函数，使用HDF5格式存储，支持流式读取
    """
    dir_path = os.path.dirname(dti_dir)
    optimized_path = f'{dir_path}/optimized_features.h5'
    
    # 检查是否已经存在优化版数据
    if os.path.exists(optimized_path):
        with h5py.File(optimized_path, 'r') as f:
            num_samples = f['protein'].shape[0]
        print(f"Optimized HDF5 data exists: {optimized_path} with {num_samples} samples")
        return {'feature_path': optimized_path, 'num_samples': num_samples}
    
    print(f"Creating optimized HDF5 data: {optimized_path}")
    print(f"Parsing {dti_dir}, {drug_dir}, {protein_dir} with length {prot_len}, type {prot_vec}")

    # 读取原始数据
    protein_col = "Protein_ID"
    drug_col = "Compound_ID"
    col_names = [protein_col, drug_col]
    if with_label:
        label_col = "Label"
        col_names += [label_col]
    
    dti_df = pd.read_csv(dti_dir)
    drug_df = pd.read_csv(drug_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")

    # 合并数据
    if prot_vec == "Convolution":
        print("Encoding protein sequences...")
        protein_df["encoded_sequence"] = protein_df.Sequence.map(lambda a: encodeSeq(a, seq_dic))
    
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)
    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True)
    
    total_samples = len(dti_df)
    print(f"Total samples: {total_samples}")
    if with_label:
        print(f"Positive data: {sum(dti_df[label_col])}")
        print(f"Negative data: {total_samples - sum(dti_df[label_col])}")

    # 使用HDF5存储，支持分块处理大文件
    with h5py.File(optimized_path, 'w') as f:
        # 创建可扩展的数据集
        if prot_vec == "Convolution":
            protein_dtype = np.int32
        else:
            protein_dtype = np.float32
            
        protein_dset = f.create_dataset('protein', 
                                       shape=(total_samples, prot_len),
                                       dtype=protein_dtype,
                                       compression='gzip',
                                       compression_opts=9)
        
        drug_dset = f.create_dataset('drug', 
                                    shape=(total_samples, drug_len),
                                    dtype=np.float32,
                                    compression='gzip', 
                                    compression_opts=9)
        
        if with_label:
            label_dset = f.create_dataset('label', 
                                         shape=(total_samples,),
                                         dtype=np.int32,
                                         compression='gzip',
                                         compression_opts=9)

        # 分块处理数据，避免内存溢出
        chunk_size = 10000
        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_df = dti_df.iloc[chunk_start:chunk_end]
            
            print(f"Processing chunk {chunk_start} to {chunk_end}")
            
            # 处理药物特征
            drug_feature_chunk = np.stack(chunk_df[drug_vec].map(lambda fp: fp.split("\t")))
            drug_dset[chunk_start:chunk_end] = drug_feature_chunk.astype(np.float32)
            
            # 处理蛋白质特征
            if prot_vec == "Convolution":
                protein_feature_chunk = sequence.pad_sequences(
                    chunk_df["encoded_sequence"].values, prot_len
                )
                protein_dset[chunk_start:chunk_end] = protein_feature_chunk.astype(np.int32)
            else:
                protein_feature_chunk = np.stack(chunk_df[prot_vec].map(lambda fp: fp.split("\t")))
                protein_dset[chunk_start:chunk_end] = protein_feature_chunk.astype(np.float32)
            
            # 处理标签
            if with_label:
                label_dset[chunk_start:chunk_end] = chunk_df[label_col].values.astype(np.int32)

    print(f"Optimized HDF5 data saved: {optimized_path}")
    return {'feature_path': optimized_path, 'num_samples': total_samples}

class BatchDataGenerator:
    """
    批处理数据生成器，确保每个batch只加载必要的数据到GPU
    """
    def __init__(self, feature_path, batch_size=32, shuffle=True, mode='train'):
        self.feature_path = feature_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        
        # 获取数据基本信息
        with h5py.File(feature_path, 'r') as f:
            self.num_samples = f['protein'].shape[0]
            self.has_labels = 'label' in f
        
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_index = 0
        
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += len(batch_indices)
        
        # 对索引进行排序，确保HDF5读取时索引是递增的
        batch_indices_sorted = np.sort(batch_indices)
        
        # 直接从HDF5读取批次数据
        with h5py.File(self.feature_path, 'r') as f:
            protein_batch = f['protein'][batch_indices_sorted].astype(np.float32)
            drug_batch = f['drug'][batch_indices_sorted].astype(np.float32)
            
            if self.has_labels:
                label_batch = f['label'][batch_indices_sorted].astype(np.float32)
                return [drug_batch, protein_batch], label_batch
            else:
                return [drug_batch, protein_batch]
    
    def get_all_labels(self):
        """获取所有标签（用于评估）"""
        with h5py.File(self.feature_path, 'r') as f:
            if 'label' in f:
                return f['label'][:].astype(np.float32)
            else:
                return None

class ValidationDataGenerator(BatchDataGenerator):
    """
    验证专用数据生成器
    """
    def __init__(self, feature_path, batch_size=32):
        super().__init__(feature_path, batch_size, shuffle=False, mode='val')

class TestDataGenerator(BatchDataGenerator):
    """
    测试专用数据生成器
    """
    def __init__(self, feature_path, batch_size=32):
        super().__init__(feature_path, batch_size, shuffle=False, mode='test')

def metrics(correct_labels, predicted_labels, predicted_scores):
    """计算评估指标"""
    ACC = accuracy_score(correct_labels, predicted_labels)
    AUC = roc_auc_score(correct_labels, predicted_scores)
    CM = confusion_matrix(correct_labels, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    F1 = 2 * Pre * Rec / (Pre + Rec) if (Pre + Rec) > 0 else 0
    MCC = matthews_corrcoef(correct_labels, predicted_labels)
    precision, recall, _ = precision_recall_curve(correct_labels, predicted_scores)
    PRC = auc(recall, precision)
    return ACC, AUC, Rec, Pre, F1, MCC, PRC

class Drug_Target_Prediction(object):
    
    def PLayer(self, size, filters, activation, initializer, regularizer_param):
        def f(input):
            model_p = Convolution1D(filters=filters, kernel_size=size, padding='same', 
                                  kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
            model_p = BatchNormalization()(model_p)
            model_p = Activation(activation)(model_p)
            return GlobalMaxPooling1D()(model_p)
        return f

    def modelv(self, dropout, drug_layers, protein_strides, filters, fc_layers, prot_vec=False, prot_len=2500,
               activation='relu', protein_layers=None, initializer="glorot_normal", drug_len=2048, drug_vec="ECFP4"):
        def return_tuple(value):
            if type(value) is int:
               return [value]
            else:
               return tuple(value)

        regularizer_param = 0.001
        input_d = Input(shape=(drug_len,))
        input_p = Input(shape=(prot_len,))
        params_dic = {"kernel_initializer": initializer,
                      "kernel_regularizer": l2(regularizer_param),
        }
        input_layer_d = input_d
        if drug_layers is not None:
            drug_layers = return_tuple(drug_layers)
            for layer_size in drug_layers:
                model_d = Dense(layer_size, **params_dic)(input_layer_d)
                model_d = BatchNormalization()(model_d)
                model_d = Activation(activation)(model_d)
                model_d = Dropout(dropout)(model_d)
                input_layer_d = model_d

        if prot_vec == "Convolution":
            model_p = Embedding(26,20, embeddings_initializer=initializer,
                              embeddings_regularizer=l2(regularizer_param))(input_p)
            model_p = SpatialDropout1D(0.2)(model_p)
            model_ps = [self.PLayer(stride_size, filters, activation, initializer, regularizer_param)(model_p) 
                       for stride_size in protein_strides]
            if len(model_ps)!=1:
                model_p = Concatenate(axis=1)(model_ps)
            else:
                model_p = model_ps[0]
        else:
            model_p = input_p

        if protein_layers:
            input_layer_p = model_p
            protein_layers = return_tuple(protein_layers)
            for protein_layer in protein_layers:
                model_p = Dense(protein_layer, **params_dic)(input_layer_p)
                model_p = BatchNormalization()(model_p)
                model_p = Activation(activation)(model_p)
                model_p = Dropout(dropout)(model_p)
                input_layer_p = model_p

        model_t = Concatenate(axis=1)([model_d,model_p])
        if fc_layers is not None:
            fc_layers = return_tuple(fc_layers)
            for fc_layer in fc_layers:
                model_t = Dense(units=fc_layer, **params_dic)(model_t)
                model_t = BatchNormalization()(model_t)
                model_t = Activation(activation)(model_t)
        model_t = Dense(1, activation='tanh', activity_regularizer=l2(regularizer_param),**params_dic)(model_t)
        model_t = Lambda(lambda x: (x+1.)/2.)(model_t)

        model_f = Model(inputs=[input_d, input_p], outputs = model_t)
        return model_f

    def __init__(self, dropout=0.2, drug_layers=(1024,512), protein_windows = (10,15,20,25), filters=64,
                 learning_rate=1e-3, decay=0.0, fc_layers=None, prot_vec=None, prot_len=2500, activation="relu",
                 drug_len=2048, drug_vec="ECFP4", protein_layers=None):
        self.__dropout = dropout
        self.__drugs_layer = drug_layers
        self.__protein_strides = protein_windows
        self.__filters = filters
        self.__fc_layers = fc_layers
        self.__learning_rate = learning_rate
        self.__prot_vec = prot_vec
        self.__prot_len = prot_len
        self.__drug_vec = drug_vec
        self.__drug_len = drug_len
        self.__activation = activation
        self.__protein_layers = protein_layers
        self.__decay = decay
        self.__model_t = self.modelv(self.__dropout, self.__drugs_layer, self.__protein_strides,
                                     self.__filters, self.__fc_layers, prot_vec=self.__prot_vec,
                                     prot_len=self.__prot_len, activation=self.__activation,
                                     protein_layers=self.__protein_layers, drug_vec=self.__drug_vec,
                                     drug_len=self.__drug_len)

        opt = Adam(lr=learning_rate, decay=self.__decay)
        self.__model_t.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        K.get_session().run(tf.global_variables_initializer())

    def optimized_fit(self, feature_path, n_epoch=10, batch_size=32, validation_data=None):
        """
        优化版训练方法，使用批处理数据生成器
        """
        # 计算steps_per_epoch
        with h5py.File(feature_path, 'r') as f:
            num_samples = f['protein'].shape[0]
        steps_per_epoch = int(np.ceil(num_samples / batch_size))
        
        callbacks = []
        validation_steps = None
        
        if validation_data:
            with h5py.File(validation_data['feature_path'], 'r') as f:
                val_samples = f['protein'].shape[0]
            validation_steps = int(np.ceil(val_samples / batch_size))
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=10, 
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=5,
                    factor=0.5,
                    min_lr=1e-7
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model_weights.h5',
                    save_best_only=True,
                    save_weights_only=True,
                    monitor='val_loss'
                )
            ]
            
            # 直接使用生成器对象，不使用包装函数
            train_generator = BatchDataGenerator(feature_path, batch_size, shuffle=True)
            val_generator = ValidationDataGenerator(validation_data['feature_path'], batch_size)
            
            # 使用 fit_generator 并指定 steps_per_epoch 和 validation_steps
            history = self.__model_t.fit_generator(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_generator,
                validation_steps=validation_steps,
                epochs=n_epoch,
                callbacks=callbacks,
                workers=0,  # 禁用多进程
                verbose=1
            )
        else:
            # 直接使用生成器对象
            train_generator = BatchDataGenerator(feature_path, batch_size, shuffle=True)
            
            # 使用 fit_generator 并指定 steps_per_epoch
            history = self.__model_t.fit_generator(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=n_epoch,
                workers=0,  # 禁用多进程
                verbose=1
            )
            
        return history

    def summary(self):
        self.__model_t.summary()

    def batch_predict(self, generator, desc="Predicting"):
        """
        使用生成器进行批量预测，避免一次性加载所有数据，并显示类似训练的进度条
        """
        all_predictions = []
        all_labels = []
        
        # 重置生成器
        generator.current_index = 0
        
        total_batches = len(generator)
        
        # 逐batch预测，显示类似训练的进度条
        for batch_idx in range(total_batches):
            try:
                batch_data = generator.__next__()
                
                if isinstance(batch_data, tuple):
                    # 有标签的情况
                    inputs, labels = batch_data
                    predictions = self.__model_t.predict(inputs, verbose=0)
                    all_predictions.extend(predictions.flatten())
                    all_labels.extend(labels)
                else:
                    # 无标签的情况
                    predictions = self.__model_t.predict(batch_data, verbose=0)
                    all_predictions.extend(predictions.flatten())
                
                # 显示类似训练的进度条
                progress = (batch_idx + 1) / total_batches * 100
                bar_length = 30
                filled_length = int(bar_length * (batch_idx + 1) // total_batches)
                bar = '=' * filled_length + '>' + '.' * (bar_length - filled_length - 1)
                
                print(f'\r{desc}: [{bar}] {progress:.1f}% ({batch_idx + 1}/{total_batches})', end='', flush=True)
                
            except StopIteration:
                break
        
        print()  # 换行
        
        return np.array(all_predictions), np.array(all_labels) if all_labels else None

    def optimized_validation(self, train_feature_path, val_feature_path, test_feature_path, 
                        output_file=None, n_epoch=10, batch_size=32):
        """
        优化版验证方法，所有数据都使用批处理加载，包含进度显示
        """
        result_dict = {
            "learning_rate": [], "batch_size": [], "dropout": [], "n_epoch": [],
            "AUC_val": [], "AUPR_val": [], "ACC_val": [], "Rec_val": [], "Pre_val": [], 
            "F1_val": [], "MCC_val": [], "AUC_test": [], "AUPR_test": [], "ACC_test": [], 
            "Rec_test": [], "Pre_test": [], "F1_test": [], "MCC_test": []
        }

        best_model_AUC = 0
        last_improve = 0

        # 计算steps_per_epoch
        with h5py.File(train_feature_path, 'r') as f:
            train_samples = f['protein'].shape[0]
        steps_per_epoch = int(np.ceil(train_samples / batch_size))

        # 创建测试生成器以获取标签
        test_generator = TestDataGenerator(test_feature_path, batch_size)
        test_true_labels = test_generator.get_all_labels()

        for epoch in range(n_epoch):
            print(f'\nEpoch {epoch+1}/{n_epoch}')
            epoch_start = time.time()
            # 直接使用生成器对象
            train_generator = BatchDataGenerator(train_feature_path, batch_size, shuffle=True)
            
            # 训练 - 使用 fit_generator 并指定 steps_per_epoch
            history = self.__model_t.fit_generator(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epoch+1,
                initial_epoch=epoch,
                verbose=1,
                workers=0  # 禁用多进程
            )

            # 验证集评估 - 使用批处理预测，显示进度条
            print("\nValidation prediction:")
            val_generator = ValidationDataGenerator(val_feature_path, batch_size)
            val_predictions, val_labels = self.batch_predict(val_generator, "Validating")
            val_predicted_labels = (val_predictions >= 0.5).astype(int)
            
            ACC_val, AUC_val, Rec_val, Pre_val, F1_val, MCC_val, PRC_val = metrics(
                val_labels, val_predicted_labels, val_predictions
            )
            epoch_time = time.time() - epoch_start
            print('Time/epoch_time', epoch_time, epoch)
            self.__model_t.save(f'{self.save_model_path[:-6]}_epo{epoch}.model')

            # 测试集评估 - 使用批处理预测，显示进度条
            print("\nTest prediction:")
            epoch_start_predict = time.time()
            test_predictions, _ = self.batch_predict(test_generator, "Testing")
            test_predicted_labels = (test_predictions >= 0.5).astype(int)

            ACC_test, AUC_test, Rec_test, Pre_test, F1_test, MCC_test, PRC_test = metrics(
                test_true_labels, test_predicted_labels, test_predictions
            )

            epoch_time_predict = time.time() - epoch_start_predict
            print('Time/predict_time', epoch_time_predict, epoch)

            # 保存最佳模型
            if AUC_val > best_model_AUC:
                last_improve = epoch
                best_model_AUC = AUC_val
                if hasattr(self, 'save_model_path'):
                    self.__model_t.save(f'{self.save_model_path}_best.model')

            # 早停检查
            if epoch - last_improve >= 10:
                print(f'\nEarly stopping at epoch: {epoch}')
                break

            # 记录结果
            result_dict["learning_rate"].append(self.__learning_rate)
            result_dict["batch_size"].append(batch_size)
            result_dict["dropout"].append(self.__dropout)
            result_dict["n_epoch"].append(epoch)
            result_dict["AUC_val"].append(AUC_val)
            result_dict["AUPR_val"].append(PRC_val)
            result_dict["ACC_val"].append(ACC_val)
            result_dict["Rec_val"].append(Rec_val)
            result_dict["Pre_val"].append(Pre_val)
            result_dict["F1_val"].append(F1_val)
            result_dict["MCC_val"].append(MCC_val)
            result_dict["AUC_test"].append(AUC_test)
            result_dict["AUPR_test"].append(PRC_test)
            result_dict["ACC_test"].append(ACC_test)
            result_dict["Rec_test"].append(Rec_test)
            result_dict["Pre_test"].append(Pre_test)
            result_dict["F1_test"].append(F1_test)
            result_dict["MCC_test"].append(MCC_test)

            print(f"\nVal - AUC: {AUC_val:.3f}, AUPR: {PRC_val:.3f}, ACC: {ACC_val:.3f}")
            print(f"Test - AUC: {AUC_test:.3f}, AUPR: {PRC_test:.3f}, ACC: {ACC_test:.3f}")

        # 保存结果
        if output_file:
            result_df = pd.DataFrame(result_dict)
            result_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")

        return result_dict

    def optimized_predict(self, feature_path, batch_size=32):
        """优化版预测方法，使用批处理生成器，显示进度条"""
        test_generator = TestDataGenerator(feature_path, batch_size)
        predictions, _ = self.batch_predict(test_generator, "Predicting")
        return predictions

    def save(self, output_file):
        self.__model_t.save(output_file)

    # def save(self, output_file):
    #     """保存模型权重"""
    #     self.__model_t.save_weights(output_file)

    # def load(self, input_file):
    #     """加载模型权重"""
    #     self.__model_t.load_weights(input_file)

# # 内存监控工具（可选）
# def get_gpu_memory_usage():
#     """获取GPU内存使用情况"""
#     try:
#         import pynvml
#         pynvml.nvmlInit()
#         handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#         info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         return info.used / 1024**3  # 返回GB
#     except:
#         return 0

if __name__ == '__main__':
    import argparse
    
    # 设置随机种子
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.set_random_seed(seed_value)

    # # 配置GPU
    # setup_gpu_memory()

    parser = argparse.ArgumentParser(description="""
    Optimized DTI Prediction Model with Batch-wise Data Loading for GPU Memory Efficiency
    """)
    
    # 训练参数
    parser.add_argument("dti_dir", help="Training DTI information [drug, target, label]")
    parser.add_argument("drug_dir", help="Training drug information [drug, SMILES,[feature_name, ..]]")
    parser.add_argument("protein_dir", help="Training protein information [protein, seq, [feature_name]]")
    
    # 测试参数
    parser.add_argument("--test-name", '-n', help="Name of test data sets")
    parser.add_argument("--test-dti-dir", "-i", help="Test dti [drug, target, [label]]")
    parser.add_argument("--test-drug-dir", "-d", help="Test drug information [drug, SMILES,[feature_name, ..]]")
    parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]")
    parser.add_argument("--with-label", "-W", help="Existence of label information in test DTI", action="store_true")
    
    # 结构参数
    parser.add_argument("--window-sizes", '-w', help="Window sizes for model", default=[10, 15, 20, 25], nargs="*", type=int)
    parser.add_argument("--protein-layers","-p", help="Dense layers for protein", default=[128, 64], nargs="*", type=int)
    parser.add_argument("--drug-layers", '-c', help="Dense layers for drugs", default=[128], nargs="*", type=int)
    parser.add_argument("--fc-layers", '-f', help="Dense layers for concatenated layers", default=[256], nargs="*", type=int)
    
    # 训练参数
    parser.add_argument("--learning-rate", '-r', help="Learning rate for training", default=1e-4, type=float)
    parser.add_argument("--n-epoch", '-e', help="Number of epochs for training", type=int, default=15)
    
    # 类型参数
    parser.add_argument("--prot-vec", "-v", help="Type of protein feature", type=str, default="Convolution")
    parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)
    parser.add_argument("--drug-vec", "-V", help="Type of drug feature", type=str, default="morgan_fp_r2")
    parser.add_argument("--drug-len", "-L", help="Drug vector length", default=2048, type=int)
    
    # 其他超参数
    parser.add_argument("--activation", "-a", help='Activation function', type=str, default='elu')
    parser.add_argument("--dropout", "-D", help="Dropout ratio", default=0.2, type=float)
    parser.add_argument("--n-filters", "-F", help="Number of filters for convolution", default=64, type=int)
    parser.add_argument("--batch-size", "-b", help="Batch size", default=32, type=int)
    parser.add_argument("--decay", "-y", help="Learning rate decay", default=1e-4, type=float)
    
    # 模式参数
    parser.add_argument("--validation", help="Execute validation", action="store_true", default=False)
    parser.add_argument("--predict", help="Predict interactions", action="store_true", default=False)
    
    # 输出参数
    parser.add_argument("--save-model", "-m", help="Save model path", type=str)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)

    args = parser.parse_args()

    # 模型参数
    model_params = {
        "drug_layers": args.drug_layers,
        "protein_windows": args.window_sizes,
        "protein_layers": args.protein_layers,
        "fc_layers": args.fc_layers,
        "learning_rate": args.learning_rate,
        "decay": args.decay,
        "activation": args.activation,
        "filters": args.n_filters,
        "dropout": args.dropout,
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
        "drug_vec": args.drug_vec,
        "drug_len": args.drug_len
    }

    print("\tOptimized Model Parameters Summary\t")
    print("=====================================================")
    for key in model_params.keys():
        print("{:20s} : {:10s}".format(key, str(model_params[key])))
    print("=====================================================")

    # 创建模型
    dti_prediction_model = Drug_Target_Prediction(**model_params)
    dti_prediction_model.summary()

    # 解析数据（使用优化版）
    type_params = {
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
        "drug_vec": args.drug_vec,
        "drug_len": args.drug_len
    }
    
    train_info = optimized_parse_data(args.dti_dir, args.drug_dir, args.protein_dir, 
                                    with_label=True, **type_params)

    # 预测模式
    if args.predict:
        print("Running prediction mode...")
        # initial_memory = get_gpu_memory_usage()
        # print(f"Initial GPU memory usage: {initial_memory:.2f} GB")
        
        dti_prediction_model.optimized_fit(
            train_info['feature_path'],
            n_epoch=args.n_epoch,
            batch_size=args.batch_size
        )
        
        # current_memory = get_gpu_memory_usage()
        # print(f"GPU memory usage after training: {current_memory:.2f} GB")
        
        if args.test_dti_dir:
            test_info = optimized_parse_data(args.test_dti_dir, args.test_drug_dir, 
                                           args.test_protein_dir, with_label=args.with_label, **type_params)
            predictions = dti_prediction_model.optimized_predict(test_info['feature_path'], args.batch_size)
            
            if args.output:
                result_df = pd.DataFrame({'predicted': predictions})
                if args.with_label:
                    test_generator = TestDataGenerator(test_info['feature_path'], args.batch_size)
                    true_labels = test_generator.get_all_labels()
                    if true_labels is not None:
                        result_df['label'] = true_labels
                result_df.to_csv(args.output, index=False)
                print(f"Predictions saved to {args.output}")

    # 验证模式
    if args.validation and args.test_dti_dir:
        print("Running validation mode...")
        val_info = optimized_parse_data(args.test_dti_dir, args.test_drug_dir, 
                                      args.test_protein_dir, with_label=True, **type_params)
        
        # 假设测试集路径格式为 validation/ 和 test/ 目录
        dir_path_val = os.path.dirname(args.test_dti_dir)
        test_dti_path = f'{dir_path_val[:-4]}/test/dti.csv'
        test_drug_path = f'{dir_path_val[:-4]}/test/compound.csv' 
        test_protein_path = f'{dir_path_val[:-4]}/test/protein.csv'
        
        test_info = optimized_parse_data(test_dti_path, test_drug_path, test_protein_path, 
                                       with_label=True, **type_params)
        
        if args.save_model:
            dti_prediction_model.save_model_path = args.save_model
            
        # 监控GPU内存
        # initial_memory = get_gpu_memory_usage()
        # print(f"Initial GPU memory usage: {initial_memory:.2f} GB")
            
        dti_prediction_model.optimized_validation(
            train_info['feature_path'],
            val_info['feature_path'], 
            test_info['feature_path'],
            output_file=args.output,
            n_epoch=args.n_epoch,
            batch_size=args.batch_size
        )
        
        # final_memory = get_gpu_memory_usage()
        # print(f"Final GPU memory usage: {final_memory:.2f} GB")
        # print(f"Memory delta: {final_memory - initial_memory:.2f} GB")

    print("Optimized training completed!")