#!/usr/bin/env python3

import os
import random
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser, MMCIFIO
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to read sequences from proteinnet7
def read_proteinnet_sequences(file_path):
    sequences = []
    with open(file_path, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            sequences.append(str(record.seq))
    return sequences

def introduce_variation(sequence, num_variants=5, mutation_rate=0.1):
    # Generate variants by introducing mutations
    variants = []
    for _ in range(num_variants):
        mutated_seq = list(sequence)
        for i in range(len(sequence)):
            if random.random() < mutation_rate:
                mutated_seq[i] = random.choice("ACDEFGHIKLMNPQRSTVWY")  # Random AA substitution
        variants.append("".join(mutated_seq))
    return variants

def save_to_mmcif(structure, output_path):
    # Save the modified structure to mmCIF format
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_path)

def generate_and_save_structures(variants, template_structure_path, output_dir):
    # Parse the template PDB structure
    parser = PDBParser(QUIET=True)
    template_structure = parser.get_structure("template", template_structure_path)
    
    # Create and save variants in mmCIF format
    for i, variant_seq in enumerate(variants):
        # Placeholder for actual structural mutation if necessary
        output_path = os.path.join(output_dir, f"variant_{i+1}.cif")
        save_to_mmcif(template_structure, output_path)
        print(f"Saved variant {i+1} structure to {output_path}")

# ML model for protein stability prediction
def extract_sequence_features(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    features = np.array([sequence.count(aa) / len(sequence) for aa in amino_acids])
    return features

def generate_training_data_from_file(file_path):
    sequences = read_proteinnet_sequences(file_path)
    data = []
    labels = []
    
    for sequence in sequences:
        features = extract_sequence_features(sequence)
        
        # Example stability score as label (replace with actual data if available)
        stability_score = random.uniform(-10, 10)  # Synthetic stability score range
        data.append(features)
        labels.append(stability_score)
    
    return np.array(data), np.array(labels)

def train_stability_model(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

def predict_stability(model, sequence):
    features = extract_sequence_features(sequence).reshape(1, -1)
    return model.predict(features)[0]

# Main function to train and use the model, then generate structural and stability data for variants
def main():
    # Read sequences from the proteinnet4 file for training
    proteinnet_file = "proteinnet7"
    X, y = generate_training_data_from_file(proteinnet_file)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train stability model
    stability_model = train_stability_model(X_train, y_train)
    y_pred = stability_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Test MSE: {mse:.3f}")

    # Generate variants for a specific sequence
    fasta_file = "mecA.fasta"
    template_structure_path = "2d45.pdb"
    # sequence = read_proteinnet_sequences(fasta_file)[0]  # Use first sequence for variants
    sequence = fasta_file
    num_variants = 5
    variants = introduce_variation(sequence, num_variants=num_variants)
    
    # Predict and display stability for each variant, and save each variant as a structure
    output_dir = "./variants"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, variant_seq in enumerate(variants):
        # Predict stability score
        stability_score = predict_stability(stability_model, variant_seq)
        print(f"Variant {i+1}: Stability Score = {stability_score:.3f}")
        
        # Generate and save structure in mmCIF format
        generate_and_save_structures([variant_seq], template_structure_path, output_dir)

# Run the main function
if __name__ == "__main__":
    main()
