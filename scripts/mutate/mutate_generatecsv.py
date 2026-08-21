import pandas as pd
import os
import sys


def generate_mutated_sequences(sequence, mutation_set='ARNDCQEGHILKMFPSTWYV'):
    """
    Generate all possible mutated protein sequences for a given sequence.

    Parameters:
        sequence (str): Original protein sequence.
        mutation_set (str): Set of possible amino acid mutations (default: 20 standard amino acids).

    Returns:
        tuple: (list of mutated sequences, list of corresponding mutation positions)
    """
    n = len(sequence)
    mutated_sequences = [sequence]
    mutated_positions = [0]

    for i in range(n):
        original_aa = sequence[i]
        for mutation in mutation_set:
            if mutation != original_aa:
                mutated_list = list(sequence)
                mutated_list[i] = mutation
                mutated_seq = ''.join(mutated_list)
                mutated_sequences.append(mutated_seq)
                mutated_positions.append(i + 1)

    return mutated_sequences, mutated_positions


def data_file_dir_csv(folder_path):
    """Return a list of CSV filenames in the given directory."""
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_dir = folder_path

    csv_files = data_file_dir_csv(folder_path)
    print("CSV files found:", csv_files)

    os.makedirs(os.path.join(output_dir, 'mutated_data'), exist_ok=True)

    for csv_file in csv_files:
        df_data = pd.read_csv(os.path.join(folder_path, csv_file))

        # Collect all rows as tuples
        tuples = []
        for _, row in df_data.iterrows():
            tuples.append((
                row['Protein'],
                row['SMILES'],
                row['PDB_ID'],
                row['Ligand_Name'],
                row['siteslabel']
            ))

        for sequence, mol, pdb_id, ligand_name, siteslabel in tuples:
            print(f"Processing: {pdb_id}_{ligand_name}")

            mutated_sequences, mutated_positions = generate_mutated_sequences(str(sequence))

            df_mutated = pd.DataFrame({
                'sequence': mutated_sequences,
                'smiles': mol,
                'label': 1,
                'mutated_position': mutated_positions,
                'PDB_ID': pdb_id,
                'Ligand_Name': ligand_name,
                'siteslabel': siteslabel
            })

            output_path = os.path.join(output_dir, 'mutated_data', f"{pdb_id}_{ligand_name}.csv")
            df_mutated.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()