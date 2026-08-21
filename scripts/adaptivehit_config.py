# adaptivehit_config.py
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import os
@dataclass
class AdaptiveHITConfig:
    """AdaptiveHIT configuration"""
    # Basic configuration
    model_names: List[str] = None
    base_predictions_dim: int = 4
    hidden_dims: List[int] = (256, 128)
    dropout: float = 0.1
    learning_rate: float = 0.0001
    batch_size: int = 128
    epochs: int = 50
    early_stopping: int = 10
    num_workers: int = 8
    seed: int = 42

    # Strategy selection (mutually exclusive)
    strategy: str = "probability_only"  # Options: "probability_only", "full_representations"

    # Representation configuration (only when strategy="full_representations")
    use_protein_representation: bool = False
    use_drug_representation: bool = False
    fusion_method: str = "concat"  # Options: "concat", "gate", "prob_attention"

    # Protein representation (ESM-2 only)
    protein_rep_type: str = "esm2"
    protein_rep_dim: int = 2560

    # Drug representation (X-Mol only)
    drug_rep_type: str = "xmollm"
    drug_rep_dim: int = 768

    # CNN configuration for protein and drug
    cnn_channels: List[int] = (64, 128, 256)
    cnn_kernel_sizes: List[int] = (3, 5, 7)
    cnn_output_dim: int = 256

    # ESM-2 encoder configuration
    esm2_hid_dim: int = 256
    esm2_n_layers: int = 3
    esm2_kernel_size: int = 9

    # X-Mol encoder configuration
    xmol_hid_dim: int = 128
    xmol_n_layers: int = 2
    xmol_kernel_size: int = 7

    # Path configuration

    adap_root = os.environ.get('ADAP_MODEL_ROOT')

    protein_emb_dir: str = ""
    drug_emb_dir: str = ""
    data_dir: str = "data/"
    data_subdir: str = ""

    # Data column configuration
    compound_id_col: str = "drugid"
    protein_id_col: str = "protid"
    smiles_col: str = "Compound_ID"
    sequences_col: str = "Protein_ID"
    label_col: str = "label_origin"

    # Unified dimensions
    unified_protein_dim: int = 256
    unified_drug_dim: int = 256

    # Fusion configuration
    fusion_hidden_dim: int = 256
    adaptivehit_hidden_dim: int = 128

    # Weight balance configuration (entropy-based only)
    use_weight_balance: bool = False
    balance_loss_weight: float = 0.5

    # Training mode (normal only)
    training_mode: str = "normal"

    # Debug mode
    debug_mode: bool = False

    # Device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Optimizer parameters
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    def __post_init__(self):
        adap_root = os.environ.get('ADAP_MODEL_ROOT')
        if adap_root is None:
            raise EnvironmentError(
                "Environment variable ADAP_MODEL_ROOT is not set. "
                "Please export it before running this script."
            )
        base_path = Path(adap_root)
        self.protein_emb_dir = str(base_path / "_ForFeatures/esm2/data/esm2_t36_3B_UR50D")
        self.drug_emb_dir = str(base_path / "_ForFeatures/xmol/FT_to_embedding/data/for_output")
        if not Path(self.protein_emb_dir).exists():
            print(f"Warning: Protein embedding directory not found: {self.protein_emb_dir}")
        if not Path(self.drug_emb_dir).exists():
            print(f"Warning: Drug embedding directory not found: {self.drug_emb_dir}")

        if self.model_names is None:
            self.model_names = ['TransformerCPI', 'DeepConv-DTI', 'ConPLex', 'DrugBAN']

        # Set related flags based on strategy
        if self.strategy == "probability_only":
            self.use_protein_representation = False
            self.use_drug_representation = False
        elif self.strategy == "full_representations":
            self.use_protein_representation = True
            self.use_drug_representation = True
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Must be: probability_only, full_representations")

        # Validate fusion method
        if self.fusion_method not in ["concat", "gate", "prob_attention"]:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}. Must be: concat, gate, prob_attention")

        if self.debug_mode:
            self.print_config_summary()

    def print_config_summary(self):
        """Print configuration summary for debugging"""
        print("\n" + "="*80)
        print(f"AdaptiveHIT Configuration Summary - Strategy: {self.strategy}")
        print("="*80)
        print(f"[Strategy] {self.strategy}")
        print(f"  ├─ Use protein representation: {self.use_protein_representation}")
        print(f"  └─ Use drug representation: {self.use_drug_representation}")

        print(f"\n[Data]")
        print(f"  ├─ Data directory: {self.data_dir}")
        print(f"  ├─ Batch size: {self.batch_size}")
        print(f"  └─ Workers: {self.num_workers}")

        print(f"\n[Model]")
        print(f"  ├─ Base models: {len(self.model_names)}")
        print(f"  ├─ Hidden dims: {self.hidden_dims}")
        print(f"  └─ Dropout: {self.dropout}")

        if self.strategy == "full_representations":
            print(f"\n[Representations]")
            print(f"  ├─ Protein type: {self.protein_rep_type} (dim: {self.protein_rep_dim})")
            print(f"  ├─ Drug type: {self.drug_rep_type} (dim: {self.drug_rep_dim})")
            print(f"  ├─ Fusion method: {self.fusion_method}")
            print(f"  └─ Protein encoder: esm2, Drug encoder: xmol")

            if self.protein_rep_type == "esm2":
                print(f"      └─ ESM-2 directory: {self.protein_emb_dir}")
            if self.drug_rep_type == "xmollm":
                print(f"      └─ X-Mol directory: {self.drug_emb_dir}")

        print(f"\n[Training]")
        print(f"  ├─ Training mode: {self.training_mode}")
        print(f"  ├─ Weight balance: {self.use_weight_balance}")
        if self.use_weight_balance:
            print(f"      └─ Balance weight: {self.balance_loss_weight}")

        print(f"\n[Device]")
        print(f"  └─ Device: {self.device}")
        print("="*80 + "\n")

    @property
    def total_base_dim(self):
        """Return base prediction dimension (number of models)"""
        return len(self.model_names)

    def get_num_workers(self):
        """Return num_workers"""
        return self.num_workers

    def get_pin_memory(self):
        """Return pin_memory setting"""
        return True