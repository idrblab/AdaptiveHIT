# adaptivehit_fusion_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ProbabilityGuidedAttention(nn.Module):
    """Probability-guided attention fusion"""
    def __init__(self, prob_dim, drug_dim, protein_dim, hidden_dim, num_heads=4, debug=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.debug = debug

        self.query_proj = nn.Linear(prob_dim, hidden_dim)
        combined_dim = drug_dim + protein_dim
        self.key_proj = nn.Linear(combined_dim, hidden_dim)
        self.value_proj = nn.Linear(combined_dim, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.value_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        if self.debug:
            print(f"[ProbabilityGuidedAttention] Initialized")
            print(f"  prob_dim: {prob_dim}, drug_dim: {drug_dim}, protein_dim: {protein_dim}")
            print(f"  hidden_dim: {hidden_dim}, num_heads: {num_heads}")

    def forward(self, prob_features, drug_features, protein_features):
        batch_size = prob_features.size(0)

        if drug_features is None:
            drug_features = torch.zeros(batch_size, 256).to(prob_features.device)
        if protein_features is None:
            protein_features = torch.zeros(batch_size, 256).to(prob_features.device)

        combined_features = torch.cat([drug_features, protein_features], dim=1)

        Q = self.query_proj(prob_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(combined_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(combined_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q = self.query_norm(Q.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim))
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_norm(K.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim))
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_norm(V.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim))
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        attended = attended.mean(dim=1)

        output = self.output_proj(attended)
        return output


class GatedFusion(nn.Module):
    """Gate-based fusion"""
    def __init__(self, drug_dim, protein_dim, base_dim, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(drug_dim + protein_dim + base_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(drug_dim + protein_dim + base_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, drug_feat, protein_feat, base_feat):
        if drug_feat is None:
            drug_feat = torch.zeros_like(base_feat[:, :1])
        if protein_feat is None:
            protein_feat = torch.zeros_like(base_feat[:, :1])

        combined = torch.cat([drug_feat, protein_feat, base_feat], dim=1)
        gate = self.gate(combined)
        transformed = self.transform(combined)
        return gate * transformed