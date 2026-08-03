# meta_inference.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import argparse
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, \
    precision_score, accuracy_score, recall_score, confusion_matrix, precision_recall_curve, auc
import sys
import time
from datetime import datetime
import pickle

train_code_dir = os.path.dirname(os.path.abspath(__file__))
if train_code_dir not in sys.path:
    sys.path.insert(0, train_code_dir)

from meta_config import MetaConfig
from meta_learner import MetaLearner
from meta_data_loader import MetaDataset, collate_fn
from torch.utils.data import DataLoader


def override_config_paths(config, input_dir, protein_emb_dir=None, drug_emb_dir=None):
    """Override configuration paths with the current environment's paths.

    protein_emb_dir/drug_emb_dir are only overridden when explicitly passed
    (e.g. via --protein_emb_dir/--drug_emb_dir) -- otherwise the checkpoint's
    own saved config (or MetaConfig's own env-var-driven defaults, see
    meta_config.py) is left alone rather than being unconditionally
    clobbered.
    """
    if protein_emb_dir is not None:
        config.protein_emb_dir = protein_emb_dir
    if drug_emb_dir is not None:
        config.drug_emb_dir = drug_emb_dir
    config.data_dir = input_dir
    return config


class EnsemblePredictor:
    """Ensemble predictor supporting multiple strategies"""
    
    def __init__(self, model_names: List[str], mode: int, models_mode: str,
                 device: torch.device = None, data_subdir: str = "alldataset",
                 protein_emb_dir: str = None, drug_emb_dir: str = None):
        self.model_names = model_names
        self.n_models = len(model_names)
        self.mode = mode
        self.models_mode = models_mode
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.meta_models = {}
        self.weights = {}
        self.data_subdir = data_subdir
        # Only overrides a loaded checkpoint's saved embedding-dir paths when
        # explicitly provided; see override_config_paths().
        self.protein_emb_dir = protein_emb_dir
        self.drug_emb_dir = drug_emb_dir
        print(f"\n{'='*60}")
        print(f"Initializing EnsemblePredictor")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Mode: {mode}, Models mode: {models_mode}")
        print(f"Data subdir: {data_subdir}")
        print(f"Base models: {model_names}")
        print(f"Number of models: {self.n_models}")
    
    def _configure_by_strategy(self, config: MetaConfig, strategy: str):
        """Configure based on strategy name"""
        if strategy.startswith('meta_prob_only_'):
            config.use_protein_representation = False
            config.use_drug_representation = False
            config.training_mode = 'normal'
            
        elif strategy.startswith('meta_full_'):
            config.use_protein_representation = True
            config.use_drug_representation = True
            config.protein_rep_type = 'esm2'
            config.drug_rep_type = 'xmollm'
            config.training_mode = 'normal'
            
            if 'prob_attention' in strategy:
                config.fusion_method = 'prob_attention'
            elif 'gate' in strategy:
                config.fusion_method = 'gate'
            else:
                config.fusion_method = 'concat'

    def load_meta_model(self, strategy: str, model_path: str, input_dir: str = None) -> bool:
        """Load meta-learner model from checkpoint"""
        try:
            print(f"\n  [Load] Loading {strategy} from {model_path}")
            
            if not os.path.exists(model_path):
                print(f"  ✗ Model file not found: {model_path}")
                return False
            
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'config' in checkpoint:
                if isinstance(checkpoint['config'], dict):
                    config_dict = checkpoint['config'].copy()
                    config_dict['model_names'] = self.model_names
                    config_dict['strategy'] = 'meta_prob_only' if 'prob_only' in strategy else 'meta_full'
                    config_dict['data_subdir'] = self.data_subdir
                    config = MetaConfig(**config_dict)
                else:
                    config = checkpoint['config']
                    config.model_names = self.model_names
                    config.data_subdir = self.data_subdir
            else:
                config_dir = os.path.dirname(model_path)
                config_path = os.path.join(config_dir, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    config_dict['model_names'] = self.model_names
                    config_dict['data_subdir'] = self.data_subdir
                    config = MetaConfig(**config_dict)
                else:
                    config = MetaConfig(model_names=self.model_names, strategy='meta_prob_only')
                    config.data_subdir = self.data_subdir

            config = override_config_paths(config, input_dir, self.protein_emb_dir, self.drug_emb_dir)
            self._configure_by_strategy(config, strategy)
            
            print(f"\n    [Config] Strategy: {config.strategy}")
            print(f"    [Config] Use protein rep: {config.use_protein_representation}")
            print(f"    [Config] Use drug rep: {config.use_drug_representation}")
            if config.strategy == "full_representations":
                print(f"    [Config] Fusion method: {config.fusion_method}")
            
            print(f"\n    [Model] Creating MetaLearner...")
            model = MetaLearner(config)
            
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            try:
                model.load_state_dict(state_dict, strict=True)
                print(f"    [Load] ✓ Strict loading succeeded!")
            except Exception as e:
                print(f"    [Load] ⚠ Strict loading failed, attempting non-strict...")
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"    Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"    Unexpected keys: {len(unexpected_keys)}")
            
            model = model.to(self.device)
            model.eval()
            
            self.meta_models[strategy] = {
                'model': model,
                'config': config,
                'path': model_path,
                'val_auc': checkpoint.get('val_auc', 0.0)
            }
            
            print(f"\n  ✓ Successfully loaded: {strategy} (val_auc={checkpoint.get('val_auc', 0.0):.4f})")
            return True
            
        except Exception as e:
            print(f"  ✗ Failed to load {strategy}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_weights(self, strategy: str, weights_path: str) -> bool:
        """Load weight file for weighted averaging"""
        try:
            if os.path.exists(weights_path):
                with open(weights_path, 'rb') as f:
                    self.weights[strategy] = pickle.load(f)
                print(f"  ✓ Loaded weights: {strategy}")
                return True
            else:
                print(f"  ✗ Weights file not found: {weights_path}")
                return False
        except Exception as e:
            print(f"  ✗ Failed to load weights {strategy}: {e}")
            return False
    
    def create_dataloader(self, data: pd.DataFrame, config: MetaConfig, batch_size: int = 512) -> DataLoader:
        """Create data loader for predictions"""
        config.data_subdir = self.data_subdir
        dataset = MetaDataset(data, self.data_subdir, config, is_train=False)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, config),
            num_workers=0
        )

    @torch.no_grad()
    def predict_meta(self, data: pd.DataFrame, strategy: str, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Make predictions using meta-learner and return weights"""
        if strategy not in self.meta_models:
            raise ValueError(f"Model {strategy} not loaded")
        
        model_info = self.meta_models[strategy]
        model = model_info['model']
        config = model_info['config']
        config.data_subdir = self.data_subdir

        loader = self.create_dataloader(data, config, batch_size)
        
        preds = []
        weights_list = []
        total_batches = len(loader)
        
        print(f"    Predicting with {strategy} ({total_batches} batches)...")
        
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue
            
            if len(batch) == 5:
                indices, prot_feats, drug_feats, base_preds, labels = batch
            else:
                print(f"    Unexpected batch format with {len(batch)} elements")
                continue
            
            prot_feats = prot_feats.to(self.device) if prot_feats is not None else None
            drug_feats = drug_feats.to(self.device) if drug_feats is not None else None
            base_preds = base_preds.to(self.device)
            
            p, w = model(prot_feats, drug_feats, base_preds)
            preds.append(p.cpu().numpy())
            weights_list.append(w.cpu().numpy())
        
        proba = np.concatenate(preds)
        weights_array = np.concatenate(weights_list) if weights_list else None
        
        return proba, (proba >= 0.5).astype(int), weights_array

    def predict_average(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Simple average ensemble prediction with equal weights"""
        scores_list = []
        for model in self.model_names:
            col = f'Predicted_scores_{model}'
            if col in data.columns:
                scores_list.append(data[col].values)
            else:
                found = False
                for c in data.columns:
                    if model.replace('-', '_') in c or model in c:
                        scores_list.append(data[c].values)
                        found = True
                        break
                if not found:
                    print(f"  Warning: Column for {model} not found, using zeros")
                    scores_list.append(np.zeros(len(data)))
        
        scores = np.mean(scores_list, axis=0)
        # Return equal weights for all samples
        equal_weights = np.ones((len(data), self.n_models)) / self.n_models
        return scores, (scores >= 0.5).astype(int), equal_weights
    
    def predict_weighted(self, data: pd.DataFrame, strategy: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Weighted average ensemble prediction with static weights"""
        if strategy not in self.weights:
            raise ValueError(f"Weights for {strategy} not loaded")
        
        static_weights = self.weights[strategy]
        scores = np.zeros(len(data))
        
        for model in self.model_names:
            weight = static_weights.get(model, 1.0 / self.n_models)
            col = f'Predicted_scores_{model}'
            if col in data.columns:
                scores += weight * data[col].values
            else:
                for c in data.columns:
                    if model.replace('-', '_') in c or model in c:
                        scores += weight * data[c].values
                        break
        
        # Create weight matrix (same weights for all samples)
        weight_matrix = np.zeros((len(data), self.n_models))
        for i, model in enumerate(self.model_names):
            weight_matrix[:, i] = static_weights.get(model, 1.0 / self.n_models)
        
        return scores, (scores >= 0.5).astype(int), weight_matrix
    
    def predict_vote_all(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """Voting: output 1 only if all models predict 1"""
        labels_list = []
        for model in self.model_names:
            col = f'label_predict_{model}'
            if col not in data.columns:
                score_col = f'Predicted_scores_{model}'
                if score_col in data.columns:
                    labels = (data[score_col].values >= 0.5).astype(int)
                else:
                    found = False
                    for c in data.columns:
                        if model.replace('-', '_') in c or model in c:
                            if 'score' in c.lower():
                                labels = (data[c].values >= 0.5).astype(int)
                                found = True
                                break
                    if not found:
                        print(f"  Warning: No predictions for {model}, using zeros")
                        labels = np.zeros(len(data))
            else:
                labels = data[col].values
            labels_list.append(labels)
        
        all_labels = np.stack(labels_list, axis=1)
        vote_labels = (np.sum(all_labels, axis=1) == self.n_models).astype(int)
        # Vote strategies don't have continuous weights
        return None, vote_labels, None

    def predict_vote_all_minus_one(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """Voting: output 1 if at least N-1 models predict 1"""
        labels_list = []
        for model in self.model_names:
            col = f'label_predict_{model}'
            if col not in data.columns:
                score_col = f'Predicted_scores_{model}'
                if score_col in data.columns:
                    labels = (data[score_col].values >= 0.5).astype(int)
                else:
                    found = False
                    for c in data.columns:
                        if model.replace('-', '_') in c or model in c:
                            if 'score' in c.lower():
                                labels = (data[c].values >= 0.5).astype(int)
                                found = True
                                break
                    if not found:
                        print(f"  Warning: No predictions for {model}, using zeros")
                        labels = np.zeros(len(data))
            else:
                labels = data[col].values
            labels_list.append(labels)
        
        all_labels = np.stack(labels_list, axis=1)
        vote_labels = (np.sum(all_labels, axis=1) >= (self.n_models - 1)).astype(int)
        return None, vote_labels, None

    def predict_all(self, data: pd.DataFrame, strategies: List[str], batch_size: int = 512) -> pd.DataFrame:
        """Make predictions for all strategies and save weights"""
        results = data.copy()
        
        print("\n" + "=" * 60)
        print("Running Predictions")
        print("=" * 60)
        
        base_strategies = []
        for model_name in self.model_names:
            base_strategy = f'base_{model_name}'
            if base_strategy in strategies:
                base_strategies.append(base_strategy)
        
        all_strategies = strategies + base_strategies
        
        for strategy in all_strategies:
            print(f"\n▶ Strategy: {strategy}")
            try:
                if strategy.startswith('base_'):
                    model_name = strategy.replace('base_', '')
                    score_col = f'Predicted_scores_{model_name}'
                    
                    if score_col in results.columns:
                        scores = results[score_col].values
                        results[f'Predicted_scores_{strategy}'] = scores
                        results[f'label_predict_{strategy}'] = (scores >= 0.5).astype(int)
                        print(f"  ✓ Added base model predictions for {model_name}")
                    else:
                        print(f"  ✗ Base model {model_name} scores not found")
                
                elif strategy == 'average':
                    proba, labels, weights = self.predict_average(data)
                    results[f'Predicted_scores_{strategy}'] = proba
                    results[f'label_predict_{strategy}'] = labels
                    
                    # Save average weights for each base model
                    if weights is not None:
                        for i, m in enumerate(self.model_names):
                            results[f'Weight_{m}_{strategy}'] = weights[:, i]
                        print(f"  ✓ Added average predictions + {len(self.model_names)} weight columns")
                    else:
                        print(f"  ✓ Added average predictions")
                
                elif strategy == 'vote-all':
                    proba, labels, weights = self.predict_vote_all(data)
                    results[f'Predicted_scores_{strategy}'] = proba if proba is not None else -1.0
                    results[f'label_predict_{strategy}'] = labels
                    print(f"  ✓ Added vote-all predictions")
                
                elif strategy == 'vote-all-1':
                    proba, labels, weights = self.predict_vote_all_minus_one(data)
                    results[f'Predicted_scores_{strategy}'] = proba if proba is not None else -1.0
                    results[f'label_predict_{strategy}'] = labels
                    print(f"  ✓ Added vote-all-1 predictions")
                
                elif strategy.startswith('weighted_'):
                    proba, labels, weights = self.predict_weighted(data, strategy)
                    results[f'Predicted_scores_{strategy}'] = proba
                    results[f'label_predict_{strategy}'] = labels
                    
                    # Save static weights for each base model
                    if weights is not None:
                        for i, m in enumerate(self.model_names):
                            results[f'Weight_{m}_{strategy}'] = weights[:, i]
                        print(f"  ✓ Added weighted predictions + {len(self.model_names)} weight columns")
                    else:
                        print(f"  ✓ Added weighted predictions")
                
                elif strategy in self.meta_models:
                    proba, labels, weights = self.predict_meta(data, strategy, batch_size)
                    results[f'Predicted_scores_{strategy}'] = proba
                    results[f'label_predict_{strategy}'] = labels
                    
                    # Save dynamic weights for each base model (per-sample)
                    if weights is not None:
                        for i, m in enumerate(self.model_names):
                            results[f'Weight_{m}_{strategy}'] = weights[:, i]
                        print(f"  ✓ Added meta predictions + {len(self.model_names)} weight columns")
                    else:
                        print(f"  ✓ Added meta predictions")
                
                else:
                    print(f"  ✗ Strategy not available")
            
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
        
        return results


def metrics_all(correct_labels, predicted_labels, predicted_scores):
    """Calculate all evaluation metrics"""
    ACC = accuracy_score(correct_labels, predicted_labels)
    
    if predicted_scores is not None and not np.all(predicted_scores == -1):
        AUC = roc_auc_score(correct_labels, predicted_scores)
        precision, recall, _ = precision_recall_curve(correct_labels, predicted_scores)
        PRC = auc(recall, precision)
    else:
        AUC = -1.0
        PRC = -1.0
    
    CM = confusion_matrix(correct_labels, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    F1 = 2 * Pre * Rec / (Pre + Rec) if (Pre + Rec) > 0 else 0
    False_Positive = FP / (FP + TN) if (FP + TN) > 0 else 0
    MCC = matthews_corrcoef(correct_labels, predicted_labels)
    
    return AUC, PRC, ACC, Rec, Pre, F1, MCC, False_Positive


def evaluate_predictions(results: pd.DataFrame, strategies: List[str], dataset_name: str,
                        mode: int, models_mode: str, base_models: List[str],
                        eval_modes: List[str]) -> List[Dict]:
    """Comprehensive evaluation of predictions"""
    if 'label_origin' not in results.columns:
        print("  Warning: 'label_origin' column not found")
        return []
    
    all_metrics = []
    
    if 'all' in eval_modes:
        print("\n  [Eval] Overall evaluation...")
        for s in strategies:
            proba_col = f'Predicted_scores_{s}'
            label_col = f'label_predict_{s}'
            
            if proba_col not in results.columns or label_col not in results.columns:
                continue
            
            try:
                proba = results[proba_col].values
                labels = results['label_origin'].values
                pred = results[label_col].values
                
                AUC, PRC, ACC, Rec, Pre, F1, MCC, FP = metrics_all(labels, pred, proba)
                
                all_metrics.append({
                    'dataset': dataset_name,
                    'mode': mode,
                    'models_mode': models_mode,
                    'strategy': s,
                    'subset': 'all',
                    'subset_name': 'Overall',
                    'n': len(proba),
                    'AUC': AUC,
                    'PRC': PRC,
                    'MCC': MCC,
                    'Accuracy': ACC,
                    'Precision': Pre,
                    'F1': F1,
                    'False_Positive': FP,
                    'Recall': Rec,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                print(f"    ✗ Error evaluating {s}: {e}")
    
    return all_metrics


def find_model_path(model_dir: str, strategy: str, mode: int, models_mode: str) -> Optional[str]:
    """Intelligently find model path"""
    # For meta models, the checkpoint is directly in the strategy directory
    path = os.path.join(model_dir, strategy, "best_model.pth")
    if os.path.exists(path):
        return path
    else:
        print(f"    ✗ Model not found: {path}")
        return None


def find_weights_path(weights_dir: str, strategy: str, mode: int, models_mode: str) -> Optional[str]:
    """Find weights file path"""
    base_strategy = strategy.replace('weighted_', '')
    
    possible_paths = [
        os.path.join(weights_dir, f'weighted_{base_strategy}_{mode}_{models_mode}.pkl'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def save_metrics_to_csv(metrics_list: List[Dict], output_file: str):
    """Save evaluation metrics to CSV (append mode)"""
    if not metrics_list:
        return
    
    df = pd.DataFrame(metrics_list)
    
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"\n  ✓ Appended {len(df)} new metrics to {output_file}")
        print(f"     Total metrics: {len(combined_df)}")
    else:
        df.to_csv(output_file, index=False)
        print(f"\n  ✓ Created new metrics file: {output_file}")


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CHECKPOINT_DIR = os.path.join(_REPO_ROOT, 'checkpoints', 'meta')
_DEFAULT_STRATEGY = 'meta_full_esm2_xmol_prob_attention'  # the shipped pretrained checkpoint


def main():
    parser = argparse.ArgumentParser(description='Ensemble Prediction for DTI')

    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing test CSV files')
    parser.add_argument('--model_dir', type=str, default=_DEFAULT_CHECKPOINT_DIR,
                       help='Directory containing trained models (default: the shipped pretrained checkpoint)')
    parser.add_argument('--weights_dir', type=str, default=_DEFAULT_CHECKPOINT_DIR,
                       help='Directory containing weight files (only needed for weighted_* strategies)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--dataset_name', type=str, default='test', help='Dataset name for evaluation')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for prediction')
    parser.add_argument('--strategies', type=str, nargs='+', default=[_DEFAULT_STRATEGY],
                       help='Strategies to evaluate (default: the shipped pretrained meta-learner checkpoint; '
                            'point --model_dir/--strategies at your own retrained checkpoint to replace it)')
    parser.add_argument('--data_subdir', type=str, default='alldataset', help='Data subdirectory for embeddings')
    parser.add_argument('--eval_base_models', action='store_true', help='Evaluate base models as well')
    parser.add_argument('--mode', type=int, default=4, help='Mode (number of models)')
    parser.add_argument('--models_mode', type=str, default='all', help='Models mode (all/de_X)')
    parser.add_argument('--base_models', type=str, nargs='+',
                       default=['TransformerCPI', 'DeepConv-DTI', 'ConPLex', 'DrugBAN'],
                       help='Base model names')
    parser.add_argument('--eval', action='store_true', help='Enable evaluation after prediction')
    parser.add_argument('--eval_modes', type=str, nargs='+', default=['all'],
                       choices=['all', 'diff', 'similarity', 'diff_similarity'],
                       help='Evaluation modes to run')
    parser.add_argument('--similarity_file', type=str, default=None,
                       help='Path to similarity classification file')
    parser.add_argument('--protein_emb_dir', type=str, default=None,
                       help='Override a loaded checkpoint\'s precomputed ESM-2 embedding directory '
                            '(default: keep the checkpoint\'s own saved path / MetaConfig default)')
    parser.add_argument('--drug_emb_dir', type=str, default=None,
                       help='Override a loaded checkpoint\'s precomputed X-Mol embedding directory '
                            '(default: keep the checkpoint\'s own saved path / MetaConfig default)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    predictor = EnsemblePredictor(args.base_models, args.mode, args.models_mode, data_subdir=args.data_subdir,
                                   protein_emb_dir=args.protein_emb_dir, drug_emb_dir=args.drug_emb_dir)
    
    args.eval_base_models_flag = args.eval_base_models
    
    if args.eval:
        test_files = [f for f in os.listdir(args.input_dir) if f.startswith('test') and f.endswith('.csv')]
    else:
        test_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
    
    test_files.sort()
    print(f"\nFound {len(test_files)} test files: {test_files}")
    
    metrics_file = os.path.join(args.output_dir, f'all_evaluations.csv')
    
    print("\n" + "=" * 60)
    print("Loading models and weights")
    print("=" * 60)
    
    for s in args.strategies:
        if s == 'average':
            print(f"\n▶ Strategy: {s} (no loading needed)")
            continue   
        elif s == 'vote-all':
            print(f"\n▶ Strategy: {s} (no loading needed)")
            continue
        elif s == 'vote-all-1':
            print(f"\n▶ Strategy: {s} (no loading needed)")
            continue
        elif s.startswith('weighted_'):
            w_path = find_weights_path(args.weights_dir, s, args.mode, args.models_mode)
            if w_path:
                predictor.load_weights(s, w_path)
            else:
                print(f"  ⚠ Weights not found for {s}")
        
        elif s.startswith('meta_'):
            model_path = find_model_path(args.model_dir, s, args.mode, args.models_mode)
            if model_path:
                predictor.load_meta_model(s, model_path, args.input_dir)
            else:
                print(f"  ⚠ Model not found for {s}")
        
        elif s.startswith('base_'):
            print(f"\n▶ Strategy: {s} (base model, no loading needed)")
        
        else:
            print(f"  ⚠ Unknown strategy type: {s}")
    
    all_predictions = []
    
    for test_file in test_files:
        print(f"\n{'='*80}")
        print(f"Processing: {test_file}")
        print(f"{'='*80}")
        
        test_path = os.path.join(args.input_dir, test_file)
        test_data = pd.read_csv(test_path)
        print(f"Test data shape: {test_data.shape}")
        
        missing_models = []
        for model in args.base_models:
            col = f'Predicted_scores_{model}'
            if col not in test_data.columns:
                found = False
                for c in test_data.columns:
                    if model.replace('-', '_') in c or model in c:
                        test_data[col] = test_data[c]
                        found = True
                        print(f"  Mapped {c} -> {col}")
                        break
                if not found:
                    missing_models.append(model)
        
        if missing_models:
            print(f"  Warning: Missing columns for models: {missing_models}")
        
        final_strategies = list(args.strategies)
        
        if args.eval_base_models_flag:
            base_strategies = [f'base_{model}' for model in args.base_models]
            for bs in base_strategies:
                if bs not in final_strategies:
                    final_strategies.append(bs)
        
        results = predictor.predict_all(test_data, final_strategies, args.batch_size)
        
        out_file = os.path.join(args.output_dir, f'{test_file}_results_{args.mode}_{args.models_mode}.csv')
        results.to_csv(out_file, index=False)
        print(f"\n  ✓ Predictions saved: {out_file}")
        all_predictions.append(out_file)
        
        if args.eval:
            print("\n" + "=" * 60)
            print("Starting Evaluation")
            print("=" * 60)
            
            dataset_id = f'{args.dataset_name}-{test_file.replace(".csv", "")}'
            metrics_list = evaluate_predictions(
                results, final_strategies, dataset_id,
                args.mode, args.models_mode,
                args.base_models, args.eval_modes
            )
            
            if metrics_list:
                save_metrics_to_csv(metrics_list, metrics_file)
                
                print("\n" + "=" * 60)
                print(f"Evaluation Results for {test_file}")
                print("=" * 60)
                df_metrics = pd.DataFrame(metrics_list)
                summary = df_metrics[['strategy', 'subset', 'AUC', 'F1', 'MCC']].copy()
                summary['AUC'] = summary['AUC'].round(4)
                summary['F1'] = summary['F1'].round(4)
                summary['MCC'] = summary['MCC'].round(4)
                print(summary.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("All tasks completed!")
    print(f"{'='*80}")
    print(f"\nPredictions saved ({len(all_predictions)} files):")
    for f in all_predictions:
        print(f"  - {f}")
    
    if args.eval and os.path.exists(metrics_file):
        df_final = pd.read_csv(metrics_file)
        print(f"\nEvaluation metrics saved: {metrics_file}")
        print(f"Total evaluation records: {len(df_final)}")
        print(f"\nStrategies evaluated: {df_final['strategy'].unique()}")
        print(f"Subsets evaluated: {df_final['subset'].unique()}")
        
        base_strategies = [s for s in df_final['strategy'].unique() if s.startswith('base_')]
        if base_strategies:
            print(f"\nBase model results:")
            for bs in base_strategies:
                bs_data = df_final[df_final['strategy'] == bs]
                if not bs_data.empty:
                    print(f"  {bs}: AUC={bs_data['AUC'].mean():.4f} ± {bs_data['AUC'].std():.4f}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()