import gc
import argparse
import os
import pandas as pd
import numpy as np
import csv
import torch
import esm
from pathlib import Path
import sys
from tqdm import tqdm
import time

prj_path = Path(__file__).parent.resolve()
sys.path.append(prj_path)

def run(start_index, esm2type, datatype, data_dir, repr_layer, hid_dim, use_gpu=True, batch_size=1):
    # 设备设置
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB")
    
    # 模型延迟到确认真的有序列要算时再加载（见下面的 all-cached 提前返回），
    # 这样在 embedding 已随仓库分发的情况下无需下载 5.4GB 的 ESM-2 权重。
    model_location = str(prj_path / 'pretrained_esm2_models' / f'{esm2type}.pt')

    # 准备数据
    # data_path = prj_path / 'data' / f'{datatype}_prots.csv'
    data_dir = Path(data_dir)
    data_path = data_dir / 'id' / f'{datatype}_prots.csv'
    print(f"Loading data from: {data_path}")
    
    _data = pd.read_csv(data_path)
    _data['length'] = _data['sequence'].str.len()
    _data = _data.sort_values(by=['length'])
    
    data = _data[['protid','sequence']].apply(lambda x: tuple(x), axis=1).values.tolist()
    
    # 找到起始位置
    start_idx = 0
    if start_index != 'head':
        for i, (protid, _) in enumerate(data):
            if protid == start_index:
                start_idx = i
                print(f"Resuming from: {start_index} (index {i})")
                break
    
    # 准备输出目录
    
    save_path = prj_path / 'data' / f'{esm2type}' / f'{datatype}' / 'token_representations'
    save_path.mkdir(parents=True, exist_ok=True)

    # 全部都已算过就直接返回，不加载模型（权重可以完全不存在）。
    todo = [pid for pid, _ in data[start_idx:] if not (save_path / f'{pid}.npy').exists()]
    if not todo:
        print(f"All {len(data) - start_idx} embeddings already present in {save_path}; nothing to do.")
        return

    print(f"Loading model: {esm2type}  ({len(todo)} sequence(s) to compute)")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_location)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    output_csv = prj_path / 'data' / f'{esm2type}' / f'{datatype}' / f'{datatype}_all-data-merge-prot.csv'
    error_log = prj_path / 'data' / f'{esm2type}' / f'{datatype}' / 'error_log.txt'
    
    # 进度条设置
    total_seqs = len(data) - start_idx
    pbar = tqdm(
        total=total_seqs,
        desc=f"Processing {esm2type}",
        unit="seq",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    # 统计信息
    stats = {
        'processed': 0,
        'failed': 0,
        'failed_ids': [],
        'start_time': time.time(),
        'last_success_time': time.time()
    }
    
    # 检查是否需要写入表头
    write_header = not output_csv.exists() or os.path.getsize(output_csv) == 0
    
    # 处理单个序列（内存不足时使用）
    def process_single(protid, sequence, local_write_header):
        """处理单个序列"""
        try:
            # 转换批次
            batch_labels, _, batch_tokens = batch_converter([(protid, sequence)])
            batch_tokens = batch_tokens.to(device)
            
            # 推理
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            
            token_reprs = results["representations"][repr_layer]
            
            # 计算序列长度
            seq_len = (batch_tokens[0] != alphabet.padding_idx).sum().item() - 2
            
            # 保存token表示
            token_data = token_reprs[0, 1:seq_len+1].cpu().numpy()
            np.save(save_path / f'{protid}.npy', token_data)
            
            # 计算并保存序列表示
            seq_repr = token_data.mean(0)
            
            # 写入CSV
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if local_write_header:
                    writer.writerow(['protid'] + [f'{esm2type}_idx{j}' for j in range(hid_dim)])
                writer.writerow([protid] + seq_repr.tolist())
            
            # 清理
            del results, token_reprs, batch_tokens
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            return True, False  # 成功，write_header已使用
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                # 记录错误
                with open(error_log, 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Out of memory for {protid} (length: {len(sequence)})\n")
                
                # 尝试更激进的内存清理
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(1)  # 给CUDA一些时间释放内存
                
                return False, local_write_header
            else:
                # 记录其他错误
                with open(error_log, 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error for {protid}: {str(e)}\n")
                return False, local_write_header
    
    # 批处理函数
    def process_batch(batch_items, local_write_header):
        """处理批次序列"""
        try:
            batch_labels, _, batch_tokens = batch_converter(batch_items)
            batch_tokens = batch_tokens.to(device)
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            
            token_reprs = results["representations"][repr_layer]
            
            # 写入CSV
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if local_write_header:
                    writer.writerow(['protid'] + [f'{esm2type}_idx{j}' for j in range(hid_dim)])
                
                for i, (protid, sequence) in enumerate(batch_items):
                    # 计算序列长度
                    seq_len = (batch_tokens[i] != alphabet.padding_idx).sum().item() - 2
                    
                    # 保存token表示
                    token_data = token_reprs[i, 1:seq_len+1].cpu().numpy()
                    np.save(save_path / f'{protid}.npy', token_data)
                    
                    # 计算并保存序列表示
                    seq_repr = token_data.mean(0)
                    writer.writerow([protid] + seq_repr.tolist())
            
            # 清理
            del results, token_reprs, batch_tokens
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            return len(batch_items), 0, [], False  # 成功，write_header已使用
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                # 批处理内存不足，降级为单序列处理
                success_count = 0
                failed_ids = []
                current_write_header = local_write_header
                
                for protid, sequence in batch_items:
                    success, current_write_header = process_single(protid, sequence, current_write_header)
                    if success:
                        success_count += 1
                    else:
                        failed_ids.append(protid)
                
                return success_count, len(batch_items) - success_count, failed_ids, current_write_header
            else:
                # 记录其他错误
                with open(error_log, 'a') as f:
                    for protid, _ in batch_items:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Batch error for {protid}: {str(e)}\n")
                
                return 0, len(batch_items), [protid for protid, _ in batch_items], local_write_header
    
    # 主处理循环
    batch_data = []
    idx = start_idx
    
    while idx < len(data):
        protid, sequence = data[idx]
        seq_len = len(sequence)

        # 已经算过的直接跳过：既让中断后可续跑，也让随仓库分发的
        # toy_dataset 预计算 embedding 无需再下载 ESM-2 权重即可使用。
        if (save_path / f'{protid}.npy').exists():
            pbar.set_postfix({'status': 'cached'})
            stats['processed'] += 1
            pbar.update(1)
            idx += 1
            continue

        # 如果序列太长，直接使用单序列处理
        if seq_len > 2000:  # 长序列阈值，可根据需要调整
            pbar.set_postfix({'status': f'Long seq ({seq_len} aa)'})
            
            success, write_header = process_single(protid, sequence, write_header)
            if success:
                stats['processed'] += 1
            else:
                stats['failed'] += 1
                stats['failed_ids'].append(protid)
            
            pbar.update(1)
            idx += 1
            continue
        
        # 添加到批处理
        batch_data.append((protid, sequence))
        
        # 检查批次是否达到大小或是最后一个序列
        if len(batch_data) >= batch_size or idx == len(data) - 1:
            pbar.set_postfix({'status': f'Batch {len(batch_data)} seqs'})
            
            success, failed, failed_ids, write_header = process_batch(batch_data, write_header)
            
            stats['processed'] += success
            stats['failed'] += failed
            stats['failed_ids'].extend(failed_ids)
            
            # 更新进度条
            pbar.update(len(batch_data))
            
            # 更新进度条描述
            elapsed = time.time() - stats['start_time']
            speed = (stats['processed'] + stats['failed']) / elapsed if elapsed > 0 else 0
            
            postfix_info = {
                'success': stats['processed'],
                'failed': stats['failed'],
                'speed': f'{speed:.1f} seq/s',
                'current': protid[:15] + ('...' if len(protid) > 15 else '')
            }
            
            if device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                postfix_info['mem'] = f'{allocated:.1f}/{reserved:.1f}GB'
            
            pbar.set_postfix(postfix_info)
            
            # 重置批次数据
            batch_data.clear()
        
        idx += 1
    
    pbar.close()
    
    # 完成统计
    total_time = time.time() - stats['start_time']
    
    print(f"\n" + "="*80)
    print("PROCESSING COMPLETED!")
    print("="*80)
    print(f"Total sequences attempted: {stats['processed'] + stats['failed']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed to process: {stats['failed']}")
    print(f"Success rate: {stats['processed']/(stats['processed']+stats['failed'])*100:.1f}%" if (stats['processed']+stats['failed']) > 0 else "Success rate: 0%")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Average speed: {(stats['processed']+stats['failed'])/total_time:.1f} seq/s" if total_time > 0 else "Average speed: 0 seq/s")
    print(f"Results saved to: {output_csv}")
    print(f"Token representations saved to: {save_path}")
    
    if stats['failed'] > 0:
        print(f"\nFailed sequences ({stats['failed']}):")
        if len(stats['failed_ids']) <= 20:
            for failed_id in stats['failed_ids']:
                print(f"  - {failed_id}")
        else:
            print(f"  First 20 failed IDs: {', '.join(stats['failed_ids'][:20])}")
            print(f"  ... and {len(stats['failed_ids']) - 20} more")
        
        failed_ids_file = prj_path / 'data' / f'{esm2type}' / f'{datatype}' / 'failed_sequences.txt'
        with open(failed_ids_file, 'w') as f:
            f.write(f"Total failed: {stats['failed']}\n")
            if (stats['processed'] + stats['failed']) > 0:
                f.write(f"Success rate: {stats['processed']/(stats['processed']+stats['failed'])*100:.1f}%\n")
            else:
                f.write("Success rate: 0%\n")
            f.write("\nFailed sequence IDs:\n")
            for failed_id in stats['failed_ids']:
                f.write(f"{failed_id}\n")
        print(f"Failed IDs saved to: {failed_ids_file}")
    
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=str, default='head')
    parser.add_argument("--esm2type", type=str, default='esm2_t33_650M_UR50D')
    parser.add_argument("--datatype", type=str, default='human')
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--repr_layer", type=int, default=33)
    parser.add_argument("--hid_dim", type=int, default=1280)
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    
    params = parser.parse_args()
    
    # 自动设置模型参数
    model_config = {
        'esm2_t33_650M_UR50D': (33, 1280),
        'esm2_t36_3B_UR50D': (36, 2560),
        'esm2_t48_15B_UR50D': (48, 5120)
    }
    
    if params.esm2type in model_config:
        if params.repr_layer == 33:  # 使用默认值，需要更新
            params.repr_layer, params.hid_dim = model_config[params.esm2type]
    
    print("Configuration:")
    for k, v in vars(params).items():
        print(f"  {k}: {v}")
    print()
    
    run(
        start_index=params.start_index,
        esm2type=params.esm2type,
        datatype=params.datatype,
        data_dir=params.data_dir,
        repr_layer=params.repr_layer,
        hid_dim=params.hid_dim,
        use_gpu=not params.no_gpu,
        batch_size=params.batch_size
    )