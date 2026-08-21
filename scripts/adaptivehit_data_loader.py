# adaptivehit_data_loader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


class ProteinDataset(Dataset):
    """Protein dataset - ESM-2 only"""
    def __init__(self, prot_ids, sequences_list, datatype, config):
        self.prot_ids = [str(pid) for pid in prot_ids]
        self.sequences_list = [str(seq) for seq in sequences_list] if sequences_list is not None else None
        self.datatype = datatype
        self.config = config

        self.cache = {}

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, idx):
        prot_id = self.prot_ids[idx]

        cache_key = f"{prot_id}_{self.config.protein_rep_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Load ESM-2 representation
        filepath = os.path.join(
            self.config.protein_emb_dir,
            self.datatype,
            "token_representations",
            f"{prot_id}.npy"
        )
        if os.path.exists(filepath):
            protein_array = np.load(filepath, mmap_mode='r')
            if len(protein_array.shape) > 1:
                protein_array = np.mean(protein_array, axis=0)
            protein_tensor = torch.from_numpy(protein_array).float()
        else:
            print(f'Creating zero protein tensor for {prot_id} - not found in {filepath}')
            protein_tensor = torch.zeros(self.config.protein_rep_dim)

        self.cache[cache_key] = protein_tensor
        return protein_tensor


class DrugDataset(Dataset):
    """Drug dataset - X-Mol only with shared global features"""

    _xmol_features = None      # Global feature array (shared across all instances)
    _global_index = None       # Global drug_id -> index mapping
    _xmol_lock = None

    def __init__(self, drug_ids, smiles_list, datatype, config):
        self.drug_ids = [str(did) for did in drug_ids]
        self.smiles_list = [str(smi) for smi in smiles_list] if smiles_list is not None else None
        self.datatype = datatype
        self.config = config

        self.cache = {}
        self._subset_index = None

        self._init_shared_xmol_features()

    def _init_shared_xmol_features(self):
        """Initialize shared X-Mol feature array"""
        import threading

        if DrugDataset._xmol_lock is None:
            DrugDataset._xmol_lock = threading.Lock()

        if DrugDataset._xmol_features is None:
            with DrugDataset._xmol_lock:
                if DrugDataset._xmol_features is None:
                    self._load_global_feature_library()

        self._build_subset_index()

        if self.config.debug_mode:
            self._verify_cached_features()

    def _load_global_feature_library(self):
        """Load global feature library (executed only once)"""
        cache_dir = os.path.join(self.config.drug_emb_dir, "xmol_shared")

        features_path = os.path.join(cache_dir, f"features_{self.datatype}.dat")
        shape_path = features_path.replace('.dat', '_shape.pkl')
        global_index_path = os.path.join(cache_dir, f"global_index_{self.datatype}.pkl")

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Global feature library not found: {features_path}")
        if not os.path.exists(shape_path):
            raise FileNotFoundError(f"Shape info file not found: {shape_path}")
        if not os.path.exists(global_index_path):
            raise FileNotFoundError(f"Global index file not found: {global_index_path}")

        with open(shape_path, 'rb') as f:
            num_samples, dim = pickle.load(f)

        DrugDataset._xmol_features = np.memmap(
            features_path, dtype='float32', mode='r',
            shape=(num_samples, dim)
        )
        print(f"\n[X-Mol Shared] Loaded global feature library: {features_path}")
        print(f"  Shape: {DrugDataset._xmol_features.shape}")

        with open(global_index_path, 'rb') as f:
            DrugDataset._global_index = pickle.load(f)
        print(f"  Global index: {len(DrugDataset._global_index)} molecules")

    def _build_subset_index(self):
        """Build subset index for current dataset"""
        unique_drug_ids = list(set(self.drug_ids))

        if self.config.debug_mode:
            print(f"\n  [Dynamic Index] Building subset index...")
            print(f"  Total samples: {len(self.drug_ids)}")
            print(f"  Unique molecules: {len(unique_drug_ids)}")

        if DrugDataset._global_index is None:
            raise RuntimeError("Global index not loaded")

        subset_index = {}
        missing_count = 0

        for drug_id in unique_drug_ids:
            if drug_id in DrugDataset._global_index:
                subset_index[drug_id] = DrugDataset._global_index[drug_id]
            else:
                subset_index[drug_id] = -1
                missing_count += 1

        self._subset_index = subset_index

        if self.config.debug_mode:
            total = len(subset_index)
            hit_count = total - missing_count
            hit_rate = (hit_count / total * 100) if total > 0 else 0
            print(f"  Subset index built: {hit_count}/{total} ({hit_rate:.1f}%) hits")

    def _verify_cached_features(self, num_samples=3):
        """Verify cached features against original data"""
        import random

        print(f"\n  [Debug] Verifying cached features...")

        unique_drug_ids = list(set(self.drug_ids))
        valid_ids = [did for did in unique_drug_ids if did in DrugDataset._global_index]

        if len(valid_ids) < num_samples:
            valid_ids = unique_drug_ids[:num_samples]

        if len(valid_ids) == 0:
            print(f"  No molecules to verify")
            return

        test_ids = random.sample(valid_ids, min(num_samples, len(valid_ids)))

        for drug_id in test_ids:
            idx = self._subset_index.get(drug_id)
            if idx is None or idx == -1:
                continue

            cached_arr = DrugDataset._xmol_features[idx]

            filepath = os.path.join(
                self.config.drug_emb_dir,
                self.datatype,
                drug_id,
                f"{drug_id}.npy"
            )

            if not os.path.exists(filepath):
                continue

            original_arr = np.load(filepath)
            if len(original_arr.shape) > 1:
                original_arr = np.mean(original_arr, axis=0)

            cached_first10 = cached_arr[:10]
            original_first10 = original_arr[:10]

            if np.allclose(cached_first10, original_first10, rtol=1e-5, atol=1e-6):
                print(f"    {drug_id}: verified")
            else:
                print(f"    {drug_id}: mismatch detected")

    def __len__(self):
        return len(self.drug_ids)

    def __getitem__(self, idx):
        drug_id = self.drug_ids[idx]

        cache_key = f"{drug_id}_{self.config.drug_rep_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if (DrugDataset._xmol_features is not None and
            self._subset_index is not None):

            feature_idx = self._subset_index.get(drug_id, -1)

            if feature_idx != -1:
                feature = DrugDataset._xmol_features[feature_idx]
                drug_tensor = torch.from_numpy(feature.copy()).float()
            else:
                print(f'Creating zero drug tensor for {drug_id} - not found in features')
                drug_tensor = torch.zeros(self.config.drug_rep_dim)

            if len(self.cache) < 10000:
                self.cache[cache_key] = drug_tensor
            return drug_tensor

        drug_tensor = torch.zeros(self.config.drug_rep_dim)
        return drug_tensor


class AdaptiveHITDataset(Dataset):
    """Meta-learning dataset"""
    def __init__(self, df, datatype, config, is_train=True):
        self.df = df
        self.datatype = datatype
        self.config = config
        self.is_train = is_train

        self.model_predictions = self.extract_predictions(df)
        self.labels = df[config.label_col].values if config.label_col in df.columns else None

        self.prot_dataset = None
        self.drug_dataset = None

        if config.use_protein_representation:
            prot_ids = df[config.protein_id_col].values
            sequences_list = df[config.sequences_col].values
            self.prot_dataset = ProteinDataset(prot_ids, sequences_list, datatype, config)

        if config.use_drug_representation:
            if config.smiles_col in df.columns:
                smiles_col = config.smiles_col
            else:
                smiles_col = config.compound_id_col

            drug_ids = df[config.compound_id_col].values
            smiles_list = df[smiles_col].values
            self.drug_dataset = DrugDataset(drug_ids, smiles_list, datatype, config)

        if config.debug_mode:
            print(f"\n[Data Loading] Dataset: {datatype}")
            print(f"  Samples: {len(self.df)}")
            print(f"  Strategy: {config.strategy}")
            print(f"  Use protein: {self.prot_dataset is not None}")
            print(f"  Use drug: {self.drug_dataset is not None}")

    def extract_predictions(self, df):
        """Extract predictions from each base model"""
        predictions = []
        for model in self.config.model_names:
            pred_cols = [
                f"Predicted_scores_{model}",
                f"Predicted_scores_{model.replace('-', '_')}",
                f"{model}_prediction",
                f"{model.replace('-', '_')}_prediction"
            ]

            found = False
            for col in pred_cols:
                if col in df.columns:
                    predictions.append(df[col].values)
                    found = True
                    break

            if not found:
                predictions.append(np.zeros(len(df)))

        return np.column_stack(predictions)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray)):
            indices = idx
            base_preds_list = []
            labels_list = []
            prot_feats_list = []
            drug_feats_list = []

            for sample_idx in indices:
                base_pred = torch.tensor(self.model_predictions[sample_idx], dtype=torch.float32)
                base_preds_list.append(base_pred)

                if self.labels is not None:
                    labels_list.append(torch.tensor(self.labels[sample_idx], dtype=torch.float32))

                if self.prot_dataset is not None:
                    try:
                        prot_feat = self.prot_dataset[sample_idx]
                        prot_feats_list.append(prot_feat)
                    except Exception as e:
                        prot_feats_list.append(None)
                else:
                    prot_feats_list.append(None)

                if self.drug_dataset is not None:
                    try:
                        drug_feat = self.drug_dataset[sample_idx]
                        drug_feats_list.append(drug_feat)
                    except Exception as e:
                        drug_feats_list.append(None)
                else:
                    drug_feats_list.append(None)

            base_preds = torch.stack(base_preds_list)

            if labels_list:
                labels = torch.stack(labels_list)
                return (indices, prot_feats_list, drug_feats_list, base_preds, labels)
            else:
                return (indices, prot_feats_list, drug_feats_list, base_preds)

        else:
            base_pred = torch.tensor(self.model_predictions[idx], dtype=torch.float32)

            prot_feat = None
            if self.prot_dataset is not None:
                try:
                    prot_feat = self.prot_dataset[idx]
                except:
                    prot_feat = None

            drug_feat = None
            if self.drug_dataset is not None:
                try:
                    drug_feat = self.drug_dataset[idx]
                except:
                    drug_feat = None

            if self.labels is not None:
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
                return (idx, prot_feat, drug_feat, base_pred, label)
            else:
                return (idx, prot_feat, drug_feat, base_pred)


def collate_fn(batch, config):
    """Batch collation function"""
    if not batch:
        print("[COLLATE] ERROR: Batch is empty!")
        return None

    batch = [b for b in batch if b is not None]
    if not batch:
        print("[COLLATE] ERROR: All items filtered out!")
        return None

    first_item = batch[0]
    if not isinstance(first_item, tuple):
        print(f"[COLLATE] ERROR: First item is not a tuple, got {type(first_item)}")
        return None

    item_length = len(first_item)

    if item_length == 5:
        all_indices = []
        all_prot_feats = []
        all_drug_feats = []
        all_base_preds = []
        all_labels = []

        for item in batch:
            if len(item) != 5:
                continue

            indices, prot_feats, drug_feats, base_preds, labels = item

            if isinstance(indices, list):
                all_indices.extend(indices)
                if prot_feats is not None and isinstance(prot_feats, list):
                    all_prot_feats.extend(prot_feats)
                else:
                    all_prot_feats.extend([None] * len(indices))

                if drug_feats is not None and isinstance(drug_feats, list):
                    all_drug_feats.extend(drug_feats)
                else:
                    all_drug_feats.extend([None] * len(indices))

                all_base_preds.append(base_preds)
                all_labels.append(labels)
            else:
                all_indices.append(indices)
                all_prot_feats.append(prot_feats)
                all_drug_feats.append(drug_feats)
                all_base_preds.append(base_preds)
                all_labels.append(labels)

        base_preds = torch.cat([p.unsqueeze(0) if p.dim() == 1 else p for p in all_base_preds], dim=0)
        labels = torch.cat([l.unsqueeze(0) if l.dim() == 0 else l for l in all_labels], dim=0)

        processed_prot_feats = None
        if any(f is not None for f in all_prot_feats):
            prot_tensors = []
            for feat in all_prot_feats:
                if feat is not None and isinstance(feat, torch.Tensor):
                    prot_tensors.append(feat)
                else:
                    prot_tensors.append(torch.zeros(config.protein_rep_dim))
            if prot_tensors:
                processed_prot_feats = torch.stack(prot_tensors)

        processed_drug_feats = None
        if any(f is not None for f in all_drug_feats):
            drug_tensors = []
            for feat in all_drug_feats:
                if feat is not None and isinstance(feat, torch.Tensor):
                    drug_tensors.append(feat)
                else:
                    drug_tensors.append(torch.zeros(config.drug_rep_dim))
            if drug_tensors:
                processed_drug_feats = torch.stack(drug_tensors)

        return all_indices, processed_prot_feats, processed_drug_feats, base_preds, labels

    elif item_length == 4:
        indices, prot_feats, drug_feats, base_preds = zip(*batch)
        indices = list(indices)
        base_preds = torch.stack(base_preds)

        processed_prot_feats = None
        if any(f is not None for f in prot_feats):
            prot_tensors = []
            for feat in prot_feats:
                if feat is not None:
                    prot_tensors.append(feat)
                else:
                    prot_tensors.append(torch.zeros(config.protein_rep_dim))
            processed_prot_feats = torch.stack(prot_tensors)

        processed_drug_feats = None
        if any(f is not None for f in drug_feats):
            drug_tensors = []
            for feat in drug_feats:
                if feat is not None:
                    drug_tensors.append(feat)
                else:
                    drug_tensors.append(torch.zeros(config.drug_rep_dim))
            processed_drug_feats = torch.stack(drug_tensors)

        return indices, processed_prot_feats, processed_drug_feats, base_preds, None

    else:
        print(f"[COLLATE] ERROR: Unexpected tuple length: {item_length}")
        return None


def create_data_loaders(train_df, val_df, test_df, config):
    """Create data loaders"""
    data_subdir = getattr(config, 'data_subdir', 'alldataset')
    train_dataset = AdaptiveHITDataset(train_df, data_subdir, config, is_train=True)
    val_dataset = AdaptiveHITDataset(val_df, data_subdir, config, is_train=False)
    test_dataset = AdaptiveHITDataset(test_df, data_subdir, config, is_train=False) if test_df is not None else None

    num_workers = config.get_num_workers()
    pin_memory = config.get_pin_memory()

    print(f"\n[DataLoader Configuration]")
    print(f"  num_workers: {num_workers}")
    print(f"  pin_memory: {pin_memory}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, config),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, config),
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, config),
        num_workers=num_workers,
        pin_memory=pin_memory
    ) if test_dataset is not None else None

    return train_loader, val_loader, test_loader