import os
import sys
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm

def build_global_xmol_cache(drug_emb_dir, datatype, output_cache_dir, all_drug_ids):
    os.makedirs(output_cache_dir, exist_ok=True)
    
    features_path = os.path.join(output_cache_dir, f"features_{datatype}.dat")
    shape_path = features_path.replace('.dat', '_shape.pkl')
    global_index_path = os.path.join(output_cache_dir, f"global_index_{datatype}.pkl")
    
    if os.path.exists(features_path) and os.path.exists(global_index_path):
        print(f"Global feature cache already exists: {features_path}")
        return
    
    print(f"\nBuilding global X-Mol feature cache...")
    print(f"  Total molecules: {len(all_drug_ids)}")
    print(f"  Feature dimension: 768")
    
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
            
            filepath = os.path.join(drug_emb_dir, datatype, drug_id, f"{drug_id}.npy")
            
            if os.path.exists(filepath):
                try:
                    arr = np.load(filepath)
                    if len(arr.shape) > 1:
                        arr = np.mean(arr, axis=0)
                    features[idx] = arr[:dim]
                    success_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to load {drug_id}: {e}")
        
        features.flush()
    
    features.flush()
    
    with open(shape_path, 'wb') as f:
        pickle.dump((num_samples, dim), f)
    
    with open(global_index_path, 'wb') as f:
        pickle.dump(global_index, f)
    
    print(f"\n✓ Build complete")
    print(f"  Feature cache: {features_path}")
    print(f"  Global index: {global_index_path}")
    print(f"  Successfully loaded: {success_count}/{num_samples} ({success_count/num_samples*100:.1f}%)")

def main():
    data_dir = sys.argv[1]
    data_subdir = sys.argv[2]
    drug_emb_dir = sys.argv[3]

    drugs_df = pd.read_csv(f"{data_dir}/id/{data_subdir}_drugs.csv")
    all_drug_ids = list(set(drugs_df['drugid'].values))

    build_global_xmol_cache(
        drug_emb_dir=drug_emb_dir,
        datatype=data_subdir,
        output_cache_dir=f"{drug_emb_dir}/xmol_shared",
        all_drug_ids=all_drug_ids
    )

if __name__ == "__main__":
    main()