import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    x = [sample for sample in x if sample is not None]
    d, p, y = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y)


def drop_oversized_molecules(df, max_drug_nodes=290, smiles_col="SMILES"):
    """Drop rows whose molecule has more heavy atoms than max_drug_nodes.

    DTIDataset.__getitem__ pads every molecule graph up to max_drug_nodes
    virtual nodes, which only works for molecules at or under that size.
    Filtering here keeps predict_*.py's per-row output CSV aligned with the
    input dataframe -- DTIDataset itself only skips these rows at batch time
    (see graph_collate_func), which would otherwise desync predicted_scores
    from df_test's row count.
    """
    from rdkit import Chem
    num_atoms = df[smiles_col].map(lambda s: Chem.MolFromSmiles(s).GetNumAtoms())
    oversized = num_atoms > max_drug_nodes
    if oversized.any():
        for smiles in df.loc[oversized, smiles_col]:
            print(f"Warning: dropping molecule with > {max_drug_nodes} atoms: {smiles}")
        df = df.loc[~oversized].reset_index(drop=True)
    return df


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
