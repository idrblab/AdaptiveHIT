"""
Single PDB-Ligand mutation analysis pipeline.
Computes average absolute differences from reference, merges external region
labels, normalizes, generates top-10% labels, and creates publication-ready plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------- Stage 1: Compute average differences --------------------
def compute_average_differences(df, models, ref_pos=0):
    """
    For each mutated position (except ref_pos), compute the mean absolute
    difference between its predicted scores and those of the reference position.
    Returns a DataFrame with columns: mutated_position, ori_value_{model}.
    """
    # Reference row
    ref_rows = df[df['mutated_position'] == ref_pos]
    if ref_rows.empty:
        raise ValueError(f"Reference position {ref_pos} not found in input CSV.")
    ref_row = ref_rows.iloc[0]

    # Collect reference scores for each model
    ref_scores = {}
    for model in models:
        col = f'Predicted_scores_{model}'
        if col in df.columns:
            ref_scores[model] = ref_row[col]
        else:
            print(f"Warning: column '{col}' missing, skipping model {model}.")
            ref_scores[model] = None

    # Iterate over all other positions
    positions = df['mutated_position'].unique()
    positions = [p for p in positions if p != ref_pos]
    positions.sort()

    rows = []
    for pos in positions:
        pos_rows = df[df['mutated_position'] == pos]
        row = {'mutated_position': pos}
        for model in models:
            col = f'Predicted_scores_{model}'
            if col in df.columns and ref_scores.get(model) is not None:
                values = pos_rows[col].tolist()
                avg_diff = np.mean([abs(v - ref_scores[model]) for v in values])
                row[f'ori_value_{model}'] = avg_diff
            else:
                row[f'ori_value_{model}'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def merge_region_info(df, base_name, region_dir):
    """
    Merge Region and ori_label_5A from an external CSV file.
    The file is expected to be named '{base_name}_region.csv' and contain
    at least columns: mutated_position, Region, ori_label_5A.
    If file is missing or columns absent, default values are used.
    """
    if region_dir is None:
        print("No region directory provided. Using default Region='d', ori_label_5A=0.")
        df['Region'] = 'd'
        df['ori_label_5A'] = 0
        return df

    region_file = os.path.join(region_dir, f"{base_name}_region.csv")
    if not os.path.exists(region_file):
        print(f"Warning: Region file {region_file} not found. Using defaults.")
        df['Region'] = 'd'
        df['ori_label_5A'] = 0
        return df

    region_df = pd.read_csv(region_file)
    required = ['mutated_position', 'Region', 'ori_label_5A']
    if not all(col in region_df.columns for col in required):
        print(f"Warning: Region file missing required columns {required}. Using defaults.")
        df['Region'] = 'd'
        df['ori_label_5A'] = 0
        return df

    # Merge on mutated_position (left join)
    merged = pd.merge(df, region_df[required], on='mutated_position', how='left')
    merged['Region'] = merged['Region'].fillna('d')
    merged['ori_label_5A'] = merged['ori_label_5A'].fillna(0).astype(int)
    return merged


# -------------------- Stage 2: Normalization, label generation, plotting --------------------
def process_and_normalize_data(df, models):
    """Log10 + min-max normalization of ori_value_{model}."""
    df_processed = df.copy()
    epsilon = 1e-10

    for model in models:
        ori_col = f'ori_value_{model}'
        log10_col = f'ori_log10_value_{model}'
        norm_col = f'ori_log10_mutated_value_{model}'

        if ori_col not in df_processed.columns:
            print(f"  Warning: {ori_col} not found, skipping model {model}")
            continue

        values_clean = pd.to_numeric(df_processed[ori_col], errors='coerce')
        valid_mask = values_clean.notna()
        valid_vals = values_clean[valid_mask].values

        if len(valid_vals) == 0:
            df_processed[log10_col] = np.nan
            df_processed[norm_col] = np.nan
            continue

        # Shift if necessary to avoid log10(<=0)
        min_val = valid_vals.min()
        if min_val <= 0:
            offset = abs(min_val) + epsilon
            valid_vals_shifted = valid_vals + offset
        else:
            valid_vals_shifted = valid_vals

        log10_vals = np.log10(valid_vals_shifted + epsilon)

        log10_full = np.full(len(df_processed), np.nan)
        log10_full[valid_mask] = log10_vals
        df_processed[log10_col] = log10_full

        # Min-max normalization of log10 values
        log_min = log10_vals.min()
        log_max = log10_vals.max()
        if log_max > log_min:
            norm_vals = (log10_vals - log_min) / (log_max - log_min)
        else:
            norm_vals = np.zeros_like(log10_vals)

        norm_full = np.full(len(df_processed), np.nan)
        norm_full[valid_mask] = norm_vals
        df_processed[norm_col] = norm_full

        print(f"  Model {model}: {len(valid_vals)} valid points, log10 range [{log_min:.4f}, {log_max:.4f}] -> normalized to [0,1]")

    return df_processed


def generate_top10_labels(df, models):
    """Label top 10% of normalized values as positive."""
    for model in models:
        val_col = f'ori_log10_mutated_value_{model}'
        label_col = f'mutated_label_{model}_top10%'
        if val_col not in df.columns:
            print(f"  Warning: {val_col} not found, skipping label generation for {model}")
            continue

        valid = df[val_col].notna()
        if valid.sum() == 0:
            df[label_col] = 0
            continue

        vals = df.loc[valid, val_col]
        threshold = np.percentile(vals, 90)   # top 10%
        df[label_col] = 0
        df.loc[valid & (df[val_col] >= threshold), label_col] = 1
        print(f"  Generated labels for {model}: {df[label_col].sum()} positive out of {len(df)} rows")
    return df


def plot_model(df, model, output_path, pdb_id):
    """Plot bar chart for a single model."""
    value_col = f'ori_log10_mutated_value_{model}'
    label_col = f'mutated_label_{model}_top10%'

    if value_col not in df.columns or label_col not in df.columns:
        print(f"  Skip {model}: missing required columns.")
        return

    valid_data = df[df[value_col].notna()]
    if len(valid_data) == 0:
        print(f"  Skip {model}: no valid values in {value_col}")
        return

    if 'seq-order' in df.columns:
        df_filtered = df[df['seq-order'].notna()].copy()
    else:
        df_filtered = df.copy()

    df_filtered = df_filtered[df_filtered[value_col].notna()]
    if len(df_filtered) == 0:
        print(f"  Skip {model}: no data after filtering")
        return

    df_sorted = df_filtered.sort_values('mutated_position').reset_index(drop=True)

    region_color_map = {
        'a': '#FF0000',
        'b': '#00008B',
        'c': '#800080'
    }
    default_color = '#DCDCDC'

    all_positions = df_sorted['mutated_position'].values
    site_colors = []
    for _, row in df_sorted.iterrows():
        if row.get('ori_label_5A', 0) == 1:
            region = row.get('Region', '')
            site_colors.append(region_color_map.get(region, default_color))
        else:
            site_colors.append(default_color)

    bar_x, bar_height, bar_colors = [], [], []
    for _, row in df_sorted.iterrows():
        true_label = row.get('ori_label_5A', 0)
        pred_label = row[label_col]
        is_tp = (true_label == 1 and pred_label == 1)
        is_fp = (true_label == 0 and pred_label == 1)
        if not (is_tp or is_fp):
            continue

        if is_fp:
            color = '#DCDCDC'
        else:
            region = row.get('Region', '')
            color = region_color_map.get(region, default_color)

        bar_x.append(row['mutated_position'])
        bar_height.append(row[value_col])
        bar_colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    if bar_x:
        ax.bar(bar_x, bar_height, color=bar_colors, width=1, zorder=2)

    y_bottom, y_top = -0.12, -0.05
    for site, color in zip(all_positions, site_colors):
        ax.plot([site, site], [y_bottom, y_top], color=color, linewidth=2.2, clip_on=False, zorder=3)

    ax.set_ylim(-0.15, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.6, zorder=5)
    ax.set_xticks(all_positions)
    ax.set_title(f'{model} - {pdb_id}', fontweight='bold')

    plt.tight_layout()
    svg_path = os.path.join(output_path, f'{model}.svg')
    plt.savefig(svg_path, format='svg', dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  Saved: {svg_path}")


def create_mutation_plots(base_name, input_folder, output_folder, models):
    """Main plotting routine for a single PDB-Ligand pair."""
    print(f"\nProcessing: {base_name}")
    input_file = os.path.join(input_folder, f"{base_name}.csv")
    if not os.path.exists(input_file):
        print(f"  Warning: {input_file} not found, skipping.")
        return

    df = pd.read_csv(input_file)
    # Ensure ori_label_5A and Region exist (should have been merged already)
    if 'ori_label_5A' not in df.columns:
        print("  Warning: ori_label_5A not found, creating dummy zeros.")
        df['ori_label_5A'] = 0
    if 'Region' not in df.columns:
        print("  Warning: Region not found, creating dummy 'd'.")
        df['Region'] = 'd'

    print("  Normalizing data...")
    df = process_and_normalize_data(df, models)
    print("  Generating top-10% labels...")
    df = generate_top10_labels(df, models)

    # Save processed CSV (optional)
    processed_path = os.path.join(input_folder, f"{base_name}_processed.csv")
    df.to_csv(processed_path, index=False)
    print(f"  Saved processed data: {processed_path}")

    plot_dir = os.path.join(output_folder, 'plot_select_opt', base_name)
    os.makedirs(plot_dir, exist_ok=True)

    for model in models:
        plot_model(df, model, plot_dir, base_name)


# -------------------- Main entry point --------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <input_csv> [output_dir] [region_dir]")
        print("  input_csv  : Path to the CSV file containing predictions.")
        print("  output_dir : Directory for outputs (default: current).")
        print("  region_dir : Directory containing the region file '{basename}_region.csv' (optional).")
        print("               If not provided, default Region='d' and ori_label_5A=0 are used.")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    region_dir = sys.argv[3] if len(sys.argv) > 3 else None
    base_name = sys.argv[4] if len(sys.argv) > 4 else os.path.splitext(os.path.basename(input_csv))[0]
    # Extract base name from input file (remove extension and any path)

    models = ['ConPLex', 'DeepConv-DTI', 'TransformerCPI', 'DrugBAN', 'adaptivehit_full_esm2_xmol_prob_attention']

    # Read raw data
    df_raw = pd.read_csv(input_csv)
    print(f"Loaded raw data: {len(df_raw)} rows from {base_name}")

    # Stage 1: Compute average differences
    print("\n--- Stage 1: Computing average differences ---")
    df_diff = compute_average_differences(df_raw, models)
    if df_diff.empty:
        print("No data computed, exiting.")
        sys.exit(1)

    # Merge Region info from external file
    print("Merging Region information...")
    df_diff = merge_region_info(df_diff, base_name, region_dir)

    # Save intermediate file
    analyse_dir = os.path.join(output_dir, 'data_analyse')
    os.makedirs(analyse_dir, exist_ok=True)
    intermediate_file = os.path.join(analyse_dir, f"{base_name}.csv")
    df_diff.to_csv(intermediate_file, index=False)
    print(f"Saved intermediate: {intermediate_file}")

    # Stage 2: Normalize, label, and plot
    print("\n--- Stage 2: Normalization, label generation and plotting ---")
    plot_output_dir = os.path.join(output_dir, 'plot_log10_end')
    os.makedirs(plot_output_dir, exist_ok=True)

    create_mutation_plots(base_name, analyse_dir, plot_output_dir, models)
    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()