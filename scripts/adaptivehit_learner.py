# adaptivehit_learner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from adaptivehit_fusion_modules import ProbabilityGuidedAttention, GatedFusion
from adaptivehit_encoder_modules import ESM2Encoder, XMolEncoder


class WeightBalanceLoss(nn.Module):
    """Entropy-based weight balance loss"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_weight = getattr(config, 'balance_loss_weight', 0.5)

        if config.debug_mode:
            print(f"[WeightBalanceLoss] Initialized: entropy mode, weight={self.base_weight}")

    def forward(self, weights):
        batch_size, num_models = weights.shape
        device = weights.device

        entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=1)
        max_entropy = torch.log(torch.tensor(num_models, dtype=torch.float32, device=device))
        balance_loss = torch.mean(max_entropy - entropy)

        weighted_loss = balance_loss * self.base_weight

        with torch.no_grad():
            avg_weights = weights.mean(dim=0).cpu().numpy()
            weight_entropy = (-torch.sum(weights * torch.log(weights + 1e-10), dim=1)).mean().item()

        stats = {
            'balance_loss': weighted_loss.item(),
            'balance_raw': balance_loss.item(),
            'balance_weight': self.base_weight,
            'weight_entropy': weight_entropy,
            'avg_weights': str(np.round(avg_weights, 3))
        }

        return weighted_loss, stats


class CombinedLoss(nn.Module):
    """Combined BCE loss with weight balance"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.debug = config.debug_mode
        self.base_bce = nn.BCELoss(reduction='none')

        self.use_balance = getattr(config, 'use_weight_balance', False)
        if self.use_balance:
            self.balance_loss = WeightBalanceLoss(config)
            if self.debug:
                print(f"[Loss] Weight balance enabled (entropy mode)")
        else:
            self.balance_loss = None

        self.bce_weight = 1.0

    def forward(self, predictions, labels, weights=None):
        pred_min = predictions.min().item()
        pred_max = predictions.max().item()

        if pred_min < 0 or pred_max > 1:
            predictions = torch.clamp(predictions, 0.0, 1.0)

        bce_per_sample = self.base_bce(predictions, labels)
        if torch.isnan(bce_per_sample).any():
            bce_per_sample = torch.nan_to_num(bce_per_sample, nan=0.0, posinf=1.0, neginf=0.0)

        bce_loss = torch.mean(bce_per_sample)

        if self.use_balance and self.balance_loss is not None and weights is not None:
            balance_loss, balance_dict = self.balance_loss(weights)
        else:
            balance_loss = torch.tensor(0.0, device=predictions.device)
            balance_dict = {}

        total_loss = self.bce_weight * bce_loss + balance_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'bce_loss': bce_loss.item(),
            'avg_prediction': predictions.mean().item(),
        }
        loss_dict.update(balance_dict)

        if self.use_balance and bce_loss.item() > 0:
            loss_dict['balance_ratio'] = balance_loss.item() / bce_loss.item()

        return total_loss, loss_dict


class AdaptiveHITModel(nn.Module):
    """Unified AdaptiveHIT"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.debug = config.debug_mode
        self.num_models = len(config.model_names)

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        print(f"\n{'='*60}")
        print(f"[AdaptiveHITModel] Initializing - Strategy: {config.strategy}")
        print(f"{'='*60}")

        self.protein_encoder = None
        self.drug_encoder = None

        if config.strategy == "full_representations":
            self._init_full_representations()
        elif config.strategy == "probability_only":
            self._init_probability_only()

        fusion_input_dim = self._calculate_fusion_input_dim()

        self.fusion_method = config.fusion_method if config.strategy == "full_representations" else "concat"
        self.fusion_module = None
        fusion_output_dim = self._init_fusion_module(fusion_input_dim)

        self.weight_predictor = self._build_adaptivehit_network(fusion_output_dim)
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[AdaptiveHITModel] Total parameters: {total_params:,}")
        print(f"{'='*60}\n")

    def _init_probability_only(self):
        print(f"[AdaptiveHITModel] Strategy: probability only")
        print(f"  Input dim: {self.num_models}")

    def _init_full_representations(self):
        print(f"[AdaptiveHITModel] Strategy: full representations (ESM-2 + X-Mol)")

        if self.config.use_protein_representation:
            self.protein_encoder = ESM2Encoder(self.config)
            print(f"  Protein encoder: ESM2Encoder -> output dim: {self.config.unified_protein_dim}")

        if self.config.use_drug_representation:
            self.drug_encoder = XMolEncoder(self.config)
            print(f"  Drug encoder: XMolEncoder -> output dim: {self.config.unified_drug_dim}")

        print(f"  Fusion method: {self.config.fusion_method}")

    def _calculate_fusion_input_dim(self):
        if self.config.strategy == "full_representations":
            dim = self.config.total_base_dim
            if self.config.use_protein_representation:
                dim += self.config.unified_protein_dim
            if self.config.use_drug_representation:
                dim += self.config.unified_drug_dim
        else:
            dim = self.num_models

        if self.debug:
            print(f"[Dimension] Fusion input dim: {dim}")
        return dim

    def _init_fusion_module(self, input_dim):
        if self.config.strategy != "full_representations":
            self.fusion = self._concat_fusion
            if self.debug:
                print(f"[Fusion] Using concat")
            return input_dim

        if self.config.fusion_method == "concat":
            self.fusion = self._concat_fusion
            output_dim = input_dim
            if self.debug:
                print(f"[Fusion] Using concat")

        elif self.config.fusion_method == "gate":
            protein_dim = self.config.unified_protein_dim if self.config.use_protein_representation else 0
            drug_dim = self.config.unified_drug_dim if self.config.use_drug_representation else 0
            self.fusion_module = GatedFusion(drug_dim, protein_dim, self.config.total_base_dim, self.config.fusion_hidden_dim)
            self.fusion = self._gate_fusion
            output_dim = self.config.fusion_hidden_dim
            if self.debug:
                print(f"[Fusion] Using gate")

        elif self.config.fusion_method == "prob_attention":
            if self.config.use_protein_representation and self.config.use_drug_representation:
                protein_dim = self.config.unified_protein_dim
                drug_dim = self.config.unified_drug_dim
                self.fusion_module = ProbabilityGuidedAttention(
                    prob_dim=self.config.total_base_dim,
                    drug_dim=drug_dim,
                    protein_dim=protein_dim,
                    hidden_dim=self.config.fusion_hidden_dim,
                    num_heads=4,
                    debug=self.debug
                )
                self.fusion = self._prob_attention_fusion
                output_dim = self.config.fusion_hidden_dim
                if self.debug:
                    print(f"[Fusion] Using prob_attention")
            else:
                print(f"[Warning] prob_attention requires both representations, falling back to concat")
                self.fusion = self._concat_fusion
                output_dim = input_dim

        else:
            self.fusion = self._concat_fusion
            output_dim = input_dim

        return output_dim

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_adaptivehit_network(self, input_dim):
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(self.config.hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            if i == len(self.config.hidden_dims) - 1:
                layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim

        output_dim = len(self.config.model_names)
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Softmax(dim=1)
        ])

        if self.debug:
            print(f"[Meta Network] {input_dim} -> {self.config.hidden_dims} -> {output_dim}")

        return nn.Sequential(*layers)

    def _concat_fusion(self, drug_feat, protein_feat, base_pred):
        features = [base_pred]
        if drug_feat is not None:
            features.append(drug_feat)
        if protein_feat is not None:
            features.append(protein_feat)
        return torch.cat(features, dim=1)

    def _gate_fusion(self, drug_feat, protein_feat, base_pred):
        return self.fusion_module(drug_feat, protein_feat, base_pred)

    def _prob_attention_fusion(self, drug_feat, protein_feat, base_pred):
        return self.fusion_module(base_pred, drug_feat, protein_feat)

    def extract_features(self, protein_feats, drug_feats, base_preds):
        original_probs = base_preds

        if self.config.strategy == "probability_only":
            return None, None, base_preds, original_probs

        elif self.config.strategy == "full_representations":
            if self.config.use_protein_representation and protein_feats is not None:
                try:
                    protein_feats = self.protein_encoder(protein_feats)
                except Exception as e:
                    protein_feats = None

            if self.config.use_drug_representation and drug_feats is not None:
                try:
                    drug_feats = self.drug_encoder(drug_feats)
                except Exception as e:
                    drug_feats = None

            return protein_feats, drug_feats, base_preds, original_probs

        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def forward(self, protein_feats, drug_feats, base_preds):
        protein_feats, drug_feats, weight_features, original_probs = self.extract_features(
            protein_feats, drug_feats, base_preds
        )

        if self.config.strategy == "full_representations":
            features_list = [weight_features]
            if self.config.use_drug_representation and drug_feats is not None:
                features_list.append(drug_feats)
            if self.config.use_protein_representation and protein_feats is not None:
                features_list.append(protein_feats)

            if self.fusion_method == "concat":
                fused = torch.cat(features_list, dim=1)
            elif self.fusion_method == "gate":
                fused = self.fusion_module(drug_feats, protein_feats, weight_features)
            elif self.fusion_method == "prob_attention":
                fused = self.fusion_module(weight_features, drug_feats, protein_feats)
            else:
                fused = torch.cat(features_list, dim=1)

        else:
            fused = weight_features

        weights = self.weight_predictor(fused)
        final_pred = torch.sum(weights * original_probs, dim=1)

        return final_pred, weights