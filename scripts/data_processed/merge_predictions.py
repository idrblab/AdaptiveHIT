"""
Merge predictions from multiple model-specific CSV files into a single integrated CSV.
Assumes all CSV files share the same basename and are keyed by (Compound_ID, Protein_ID).
"""

import pandas as pd
import os
import argparse
from pathlib import Path


def merge_models(integrated_dir: str, model_root: str, model_names: list, output_dir: str) -> None:
    """
    Perform left-join merge of each model's prediction columns into the base integrated CSV.

    Args:
        integrated_dir: Directory containing the existing integrated CSV files.
        model_root: Parent directory where each model's subfolder (named after the model) resides.
        model_names: List of model names to be merged.
        output_dir: Destination directory for merged CSV files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_files = [f for f in os.listdir(integrated_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        print("Warning: No CSV files found in integrated directory.")
        return

    for filename in csv_files:
        integrated_path = os.path.join(integrated_dir, filename)
        try:
            df_base = pd.read_csv(integrated_path)
        except Exception as e:
            print(f"Error reading {integrated_path}: {e}")
            continue

        df_merged = df_base.copy()

        for model_name in model_names:
            model_file = os.path.join(model_root, model_name, filename)
            if not os.path.isfile(model_file):
                print(f"Warning: {model_file} not found, skipping model '{model_name}'")
                continue

            try:
                df_model = pd.read_csv(model_file)
            except Exception as e:
                print(f"Error reading {model_file}: {e}")
                continue

            score_col = f"Predicted_scores_{model_name}"
            label_col = f"label_predict_{model_name}"
            required = ['Compound_ID', 'Protein_ID', score_col, label_col]

            if not all(col in df_model.columns for col in required):
                print(f"Warning: {model_file} missing required columns, skipping")
                continue

            df_sub = df_model[required]
            df_merged = df_merged.merge(
                df_sub,
                on=['Compound_ID', 'Protein_ID'],
                how='left',
                suffixes=('', f'_{model_name}')
            )

        out_path = os.path.join(output_dir, filename)
        try:
            df_merged.to_csv(out_path, index=False)
            print(f"Saved merged file: {out_path}")
        except Exception as e:
            print(f"Error saving {out_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge additional model predictions into integrated CSV files.")
    parser.add_argument('--integrated_dir', required=True,
                        help="Path to folder with existing integrated CSV files.")
    parser.add_argument('--model_root', required=True,
                        help="Root folder containing subfolders for each model (subfolder name = model name).")
    parser.add_argument('--model_names', required=True,
                        help="Comma-separated list of model names to merge, e.g., 'TransformerCPI2,ModelX'.")
    parser.add_argument('--output_dir', required=True,
                        help="Output folder for merged CSV files.")
    args = parser.parse_args()

    model_list = [name.strip() for name in args.model_names.split(',') if name.strip()]
    if not model_list:
        print("Error: No model names provided.")
        exit(1)

    merge_models(args.integrated_dir, args.model_root, model_list, args.output_dir)