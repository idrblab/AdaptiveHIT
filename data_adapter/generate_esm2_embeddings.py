import os

if __name__ == "__main__":
    # Batch-embed a CSV of proteins into the per-protein .npy layout
    # meta_data_loader.py expects: {output_dir}/{datatype}/token_representations/{protid}.npy
    import argparse
    import numpy as np
    import pandas as pd
    import torch
    import esm

    parser = argparse.ArgumentParser(
        description='Batch ESM-2 embedding generation using the standard `fair-esm` '
                    'package (esm2_t36_3B_UR50D, 2560-dim per-residue representations -- '
                    'the variant AdaptiveHIT\'s shipped MetaLearner checkpoint expects)')
    parser.add_argument('--prots_csv', type=str, required=True,
                        help='CSV with a protid column and a sequence column '
                             '(sequence/Sequence/Protein_ID, e.g. data/toy_dataset/id/*_prots.csv)')
    parser.add_argument('--datatype', type=str, required=True,
                        help='Subdirectory name under --output_dir (matches meta_config.data_subdir)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root protein_emb_dir -- writes {output_dir}/{datatype}/token_representations/{protid}.npy')
    args = parser.parse_args()

    df = pd.read_csv(args.prots_csv)
    seq_col = next((c for c in ['sequence', 'Sequence', 'Protein_ID'] if c in df.columns), None)
    if seq_col is None:
        raise ValueError(f"Could not find a sequence column in {args.prots_csv} (columns: {list(df.columns)})")

    out_dir = os.path.join(args.output_dir, args.datatype, "token_representations")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading esm2_t36_3B_UR50D (~5.4GB, downloaded once and cached under ~/.cache/torch/hub/checkpoints/)...")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    n_ok, n_fail = 0, 0
    for _, row in df.iterrows():
        protid, sequence = str(row['protid']), str(row[seq_col])
        npy_path = os.path.join(out_dir, f"{protid}.npy")
        if os.path.exists(npy_path):
            continue
        try:
            _, _, tokens = batch_converter([(protid, sequence)])
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[36], return_contacts=False)
            # strip the BOS/EOS tokens batch_converter adds -> (seq_len, 2560) per-residue array
            rep = out["representations"][36][0, 1:len(sequence) + 1].cpu().numpy()
            np.save(npy_path, rep)
            n_ok += 1
        except Exception as e:
            print(f"  Warning: failed to embed {protid}: {e}")
            n_fail += 1
    print(f"Done: {n_ok} embedded, {n_fail} failed, out of {len(df)}")
