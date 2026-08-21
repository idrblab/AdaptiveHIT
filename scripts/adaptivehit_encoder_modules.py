# adaptivehit_encoder_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESM2Encoder(nn.Module):
    """ESM-2 protein representation encoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_dim = 2560
        self.hid_dim = config.esm2_hid_dim
        self.n_layers = config.esm2_n_layers
        self.kernel_size = config.esm2_kernel_size
        self.dropout = config.dropout
        self.device = config.device

        self.encoder = ProteinEncoder(
            protein_dim=self.input_dim,
            hid_dim=self.hid_dim,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            device=self.device
        )

        self.output_proj = nn.Linear(self.hid_dim, config.unified_protein_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if config.debug_mode:
            print(f"[ESM2Encoder] Initialized")
            print(f"  Input dim: {self.input_dim} -> Hidden: {self.hid_dim} -> Output: {config.unified_protein_dim}")

    def forward(self, protein):
        if protein.dim() == 2:
            protein = protein.unsqueeze(1)

        encoded = self.encoder(protein)
        encoded = encoded.permute(0, 2, 1)
        pooled = self.global_pool(encoded).squeeze(-1)
        output = self.output_proj(pooled)

        return output


class XMolEncoder(nn.Module):
    """X-Mol drug representation encoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_dim = 768
        self.hid_dim = config.xmol_hid_dim
        self.n_layers = config.xmol_n_layers
        self.kernel_size = config.xmol_kernel_size
        self.dropout = config.dropout
        self.device = config.device

        self.fc = nn.Linear(self.input_dim, self.hid_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.hid_dim,
                out_channels=2 * self.hid_dim,
                kernel_size=self.kernel_size,
                padding=(self.kernel_size - 1) // 2
            ) for _ in range(self.n_layers)
        ])

        self.layer_norm = nn.LayerNorm(self.hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(self.hid_dim, config.unified_drug_dim)

        if config.debug_mode:
            print(f"[XMolEncoder] Initialized")
            print(f"  Input dim: {self.input_dim} -> Hidden: {self.hid_dim} -> Output: {config.unified_drug_dim}")

    def forward(self, xmol_features):
        if xmol_features.dim() == 2:
            xmol_features = xmol_features.unsqueeze(1)

        conv_input = self.fc(xmol_features)
        conv_input = conv_input.permute(0, 2, 1)

        for conv in self.convs:
            conved = conv(self.dropout_layer(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        pooled = self.global_pool(conv_input).squeeze(-1)
        output = self.output_proj(pooled)

        return output


class ProteinEncoder(nn.Module):
    """Protein sequence encoder (based on DeepConv-DTI example)"""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([
            nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(self.n_layers)
        ])

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ln = nn.LayerNorm(hid_dim)

        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout_layer(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)

        return conved