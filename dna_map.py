import numpy as np

def read_dna_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:

            sequence = list(line.strip())
            sequences.append(sequence)
    return sequences


def map_to_indices(sequences):
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mapped_sequences = []
    for sequence in sequences:
        mapped_sequence = [base_to_index[base] for base in sequence]
        mapped_sequences.append(mapped_sequence)
    return np.array(mapped_sequences)






