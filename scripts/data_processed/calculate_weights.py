# static_weight_calculator.py
"""
Static weight exploration module - Meta-learning with weight balancing
Only retains: Logistic regression with weight balancing
"""

import numpy as np
import pandas as pd
import pickle
import os
import argparse
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class WeightCalculator:
    """Weight calculator - Meta-learning with weight balancing (logistic regression only)"""
    
    def __init__(self, model_names: List[str]):
        self.models = model_names
        self.n = len(model_names)
    
    def _get_pred_matrix(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get prediction matrix and labels"""
        preds = np.column_stack([data[f"Predicted_scores_{m}"] for m in self.models])
        labels = data['label_origin'].values
        return preds, labels
    
    def _eval_weights(self, preds: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        """Evaluate AUC for given weights"""
        return roc_auc_score(labels, np.dot(preds, weights))
    
    def _balance_penalty(self, weights: np.ndarray, mode: str = 'entropy', 
                         max_threshold: float = 0.6) -> float:
        """Calculate weight balancing penalty term"""
        weights = np.abs(weights) / (np.sum(np.abs(weights)) + 1e-10)
        
        if mode == 'entropy':
            entropy = -np.sum(weights * np.log(weights + 1e-10))
            max_entropy = np.log(self.n)
            return max_entropy - entropy
        
        elif mode == 'kl':
            uniform = np.ones(self.n) / self.n
            return np.sum(weights * np.log(weights / (uniform + 1e-10) + 1e-10))
        
        elif mode == 'concentration':
            max_w = np.max(weights)
            return max(0, max_w - max_threshold) ** 2
        
        elif mode == 'variance':
            return np.var(weights)
        
        else:
            return 0.0
    
    # ==================== Logistic Regression with Weight Balancing ====================
    def logistic_with_balance(self, data: pd.DataFrame, 
                               balance_weight: float = 0.5,
                               balance_mode: str = 'entropy',
                               max_threshold: float = 0.6,
                               penalty: str = 'l2', 
                               C: float = 1.0) -> Dict[str, float]:
        """
        Logistic regression with weight balancing
        
        Args:
            data: Validation dataset
            balance_weight: Weight for balance loss
            balance_mode: Balance mode ('entropy', 'kl', 'concentration', 'variance')
            max_threshold: Max weight threshold for concentration mode
            penalty: Regularization type
            C: Regularization strength
        """
        preds, labels = self._get_pred_matrix(data)
        
        # Negative log-likelihood loss for logistic regression
        def logistic_loss(weights):
            weights = np.abs(weights) / (np.sum(np.abs(weights)) + 1e-10)
            linear_comb = np.dot(preds, weights)
            linear_comb = np.clip(linear_comb, -50, 50)
            eps = 1e-10
            loss = -np.mean(labels * np.log(linear_comb + eps) + 
                           (1 - labels) * np.log(1 - linear_comb + eps))
            return loss
        
        # Weight balancing penalty
        def balance_penalty(weights):
            return self._balance_penalty(weights, balance_mode, max_threshold)
        
        # Total loss
        def total_loss(weights):
            return logistic_loss(weights) + balance_weight * balance_penalty(weights)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1}]
        bounds = [(0, 1) for _ in range(self.n)]
        
        # Initial weights
        init_weights = np.ones(self.n) / self.n
        
        # Optimization
        result = minimize(
            total_loss, init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-8, 'maxiter': 1000}
        )
        
        # Get optimal weights
        optimal_weights = np.abs(result.x) / (np.sum(np.abs(result.x)) + 1e-10)
        
        # Evaluation
        final_auc = self._eval_weights(preds, labels, optimal_weights)
        entropy = -np.sum(optimal_weights * np.log(optimal_weights + 1e-10))
        max_entropy = np.log(self.n)
        max_weight = np.max(optimal_weights)
        
        print(f"  Balanced logistic regression AUC: {final_auc:.4f}")
        print(f"  Weight entropy: {entropy:.4f} / {max_entropy:.4f}")
        print(f"  Max weight: {max_weight:.4f}")
        print(f"  Balance weight: {balance_weight}, Mode: {balance_mode}")
        
        return dict(zip(self.models, optimal_weights))
    
    # ==================== Save/Load ====================
    def save(self, weights: Dict, path: str):
        """Save weights to file"""
        with open(path, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Saved: {path}")
    
    def load(self, path: str) -> Dict:
        """Load weights from file"""
        with open(path, 'rb') as f:
            return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='Static weight calculator - Logistic regression with weight balancing')
    
    # Data parameters
    parser.add_argument('--input_dir', required=True, help='Data directory containing val.csv')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    
    # Weight balancing parameters
    parser.add_argument('--balance_weight', type=float, default=0.5, 
                       help='Weight for balance loss')
    parser.add_argument('--balance_mode', type=str, default='entropy',
                       choices=['entropy', 'kl', 'concentration', 'variance'],
                       help='Balance mode')
    parser.add_argument('--max_weight_threshold', type=float, default=0.6,
                       help='Max weight threshold for concentration mode')
    
    # Model combination parameters
    parser.add_argument('--skip_combinations', action='store_true',
                       help='Skip predefined combinations, use all models')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model combinations
    modes = [
        (4, ['TransformerCPI', 'DeepConv-DTI', 'ConPLex', 'DrugBAN'], 'all'),
        ]
    
    # Load data
    print("Loading data...")
    val_data = pd.read_csv(os.path.join(args.input_dir, 'val.csv'))
    print(f"Validation set: {val_data.shape}\n")
    
    for n, models, name in modes:
        print(f"\n{'='*60}")
        print(f"Model combination: {name} (n={n})")
        print(f"Models: {models}")
        print(f"{'='*60}")
        
        calc = WeightCalculator(models)
        
        # Calculate balanced logistic regression weights
        w = calc.logistic_with_balance(
            val_data,
            balance_weight=args.balance_weight,
            balance_mode=args.balance_mode,
            max_threshold=args.max_weight_threshold
        )
        
        # Evaluate
        preds, labels = calc._get_pred_matrix(val_data)
        auc = calc._eval_weights(preds, labels, list(w.values()))
        
        # Print results
        print(f"\nResults:")
        print(f"  AUC: {auc:.4f}")
        entropy = -np.sum(np.array(list(w.values())) * np.log(np.array(list(w.values())) + 1e-10))
        max_entropy = np.log(n)
        print(f"  Entropy: {entropy:.4f} / {max_entropy:.4f}")
        
        # Save weights
        save_path = os.path.join(args.output_dir, f"weighted_logistic_balanced_{n}_{name}.pkl")
        calc.save(w, save_path)
    
    print(f"\nDone! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()