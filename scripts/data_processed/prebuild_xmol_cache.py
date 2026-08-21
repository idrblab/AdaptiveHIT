# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm

def build_global_xmol_cache(drug_emb_dir, datatype, output_cache_dir, all_drug_ids):
    if not os.path.exists(output_cache_dir):
        os.makedirs(output_cache_dir)
    
    features_path = os.path.join(output_cache_dir, "features_{}.dat".format(datatype))
    shape_path = features_path.replace('.dat', '_shape.pkl')
    global_index_path = os.path.join(output_cache_dir, "global_index_{}.pkl".format(datatype))
    
    if os.path.exists(features_path) and os.path.exists(global_index_path):
        print("Global feature cache already exists: {}".format(features_path))
        return
    
    print("\nBuilding global X-Mol feature cache...")
    print("  Total molecules: {}".format(len(all_drug_ids)))
    print("  Feature dimension: 768")
    
    num_samples = len(all_drug_ids)
    dim = 768
    
    features = np.memmap(features_path, dtype='float32', mode='w+', shape=(num_samples, dim))
    global_index = {}
    
    batch_size = 10000
    success_count = 0
    
    for start in tqdm(range(0, num_samples, batch_size), desc="Loading features"):
        end = min(start + batch_size, num_samples)
        
        for i, idx in enumerate(range(start, end)):
            drug_id = all_drug_ids[idx]
            global_index[drug_id] = idx
            
            filepath = os.path.join(drug_emb_dir, datatype, drug_id, "{}.npy".format(drug_id))
            
            if os.path.exists(filepath):
                try:
                    arr = np.load(filepath)
                    if len(arr.shape) > 1:
                        arr = np.mean(arr, axis=0)
                    features[idx] = arr[:dim]
                    success_count += 1
                except Exception as e:
                    print("  Warning: Failed to load {}: {}".format(drug_id, e))
        
        features.flush()
    
    features.flush()
    
    with open(shape_path, 'wb') as f:
        pickle.dump((num_samples, dim), f)
    
    with open(global_index_path, 'wb') as f:
        pickle.dump(global_index, f)
    
    print("\n✓ Build complete")
    print("  Feature cache: {}".format(features_path))
    print("  Global index: {}".format(global_index_path))
    print("  Successfully loaded: {}/{} ({:.1f}%)".format(success_count, num_samples, success_count/num_samples*100))

def main():
    data_dir = sys.argv[1]
    data_subdir = sys.argv[2]
    drug_emb_dir = sys.argv[3]

    drugs_df = pd.read_csv("{}/id/{}_drugs.csv".format(data_dir, data_subdir))
    all_drug_ids = list(set(drugs_df['drugid'].values))

    build_global_xmol_cache(
        drug_emb_dir=drug_emb_dir,
        datatype=data_subdir,
        output_cache_dir="{}/xmol_shared".format(drug_emb_dir),
        all_drug_ids=all_drug_ids
    )

if __name__ == "__main__":
    main()