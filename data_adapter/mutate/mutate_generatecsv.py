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
        list: A list of tuples where each tuple contains:
              (position, original_amino_acid, mutated_amino_acid, mutated_sequence)
    """
    n = len(sequence)  # Length of the sequence
    mutated_sequences = []
    mutated_positions = []
    mutated_sequences.append(sequence)
    mutated_positions.append(0)

    for i in range(n):
        original_amino_acid = sequence[i]  # Original amino acid at position i
        for mutation in mutation_set:
            if mutation != original_amino_acid:  # Skip if mutation is the same as the original
                mutated_sequence = list(sequence)  # Convert sequence to a list for mutation
                mutated_sequence[i] = mutation  # Mutate the amino acid at position i
                mutated_sequence = ''.join(mutated_sequence)  # Convert back to string
                # mutations.append((i, original_amino_acid, mutation, mutated_sequence))
                mutated_sequences.append(mutated_sequence)
                mutated_positions.append(i+1)

    return mutated_sequences, mutated_positions

def data_file_dir_csv(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    return csv_files

folder_path = sys.argv[1]
output_dir = folder_path

csv_files = data_file_dir_csv(folder_path)
print(csv_files)
for data_dir in csv_files:
    # Example usage
    # data_dir = r"D:\.destbook\IDRB_AI\CPI\INtergratedCPI\mission_sub\可解释性分析\test_pdb.csv"
    # output_dir = r'D:\.destbook\IDRB_AI\CPI\INtergratedCPI\mission_sub\可解释性分析'
    df_data = pd.read_csv(fr'{folder_path}/{data_dir}')
    # df_data = df_data[:1]
    os.makedirs(fr'{output_dir}/mutated_data',exist_ok=True)
    # 遍历 DataFrame 并形成元组
    tuples = []
    for index, row in df_data.iterrows():
        # 将每一行的两个元素作为元组添加到列表
        tuples.append((row['Protein_ID'], row['Compound_ID'], row['PDB_ID'], row['Ligand_Name'], row['siteslabel']))
    print(tuples)
    for (Sequence, mol, PDB_ID, Ligand_Name, siteslabel) in tuples:
        print(PDB_ID, Ligand_Name)
        # 调用函数生成突变序列
        mutated_sequences, mutated_positions = generate_mutated_sequences(str(Sequence))
        df_mutated_data = pd.DataFrame()
        df_mutated_data['sequence'] = mutated_sequences
        df_mutated_data['smiles'] = mol
        df_mutated_data['label'] = 1
        df_mutated_data['mutated_position'] = mutated_positions
        df_mutated_data['PDB_ID'] = PDB_ID
        df_mutated_data['Ligand_Name'] = Ligand_Name
        df_mutated_data['siteslabel'] = siteslabel
        df_mutated_data.to_csv(fr'{output_dir}/mutated_data/{PDB_ID}_{Ligand_Name}.csv')


