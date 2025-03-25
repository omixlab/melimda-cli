import argparse
import os
import pandas as pd
from rdkit import Chem
import logging
from macaw import MolecularAutoencoder

def main():
    parser = argparse.ArgumentParser(description='Preprocess data for melimda')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting preprocessing...")

    if not os.path.exists(args.input):
        logging.error(f"Input file {args.input} does not exist.")
        return

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logging.info(f"Created output directory: {args.output}")

    try:
        df = pd.read_csv(args.input)
        logging.info(f"Read {len(df)} rows from {args.input}")
    except Exception as e:
        logging.error(f"Error reading input file: {str(e)}")
        return

    required_columns = ['ligand', 'id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {', '.join(missing_columns)}")
        return

    def generate_smiles(ligand_pdb):
        mol = Chem.MolFromPDBFile(ligand_pdb)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        return None

    for index, row in df.iterrows():
        ligand_path = row['ligand']
        complex_id = row['id']
        complex_dir = os.path.join(args.output, complex_id)
        ligand_pdb = os.path.join(complex_dir, "ligand.pdb")

        if os.path.exists(ligand_pdb):
            smiles = generate_smiles(ligand_pdb)
            if smiles:
                smiles_file = os.path.join(complex_dir, 'smiles.txt')
                with open(smiles_file, 'w') as f:
                    f.write(smiles)
                logging.info(f"SMILES saved for complex {complex_id}")
            else:
                logging.warning(f"Failed to generate SMILES for complex {complex_id}")

    # -------------------- MACAW Integration -------------------- #
    logging.info("Starting MACAW embedding generation...")
    model = MolecularAutoencoder()

    try:
        embeddings = []
        for index, row in df.iterrows():
            ligand_path = os.path.join(args.output, row['id'], 'ligand.pdb')
            if os.path.exists(ligand_path):
                mol = Chem.MolFromPDBFile(ligand_path)
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    embedding = model.encode([smiles])[0]
                    embeddings.append(embedding)
                else:
                    embeddings.append([0] * model.latent_dim)  # Embedding vazio em caso de falha
            else:
                embeddings.append([0] * model.latent_dim)

        embedding_df = pd.DataFrame(embeddings, columns=[f"Embedding_{i}" for i in range(model.latent_dim)])
        df = pd.concat([df, embedding_df], axis=1)

        logging.info("MACAW embeddings generated successfully.")
    except Exception as e:
        logging.error(f"Error generating MACAW embeddings: {str(e)}")
        return

    output_csv = os.path.join(args.output, 'processed_data_with_smiles.csv')
    df.to_csv(output_csv, index=False)
    logging.info(f"Processed data saved to {output_csv}")

if __name__ == "__main__":
    main()
