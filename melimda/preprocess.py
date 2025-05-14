import argparse
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import subprocess
import logging
from deepchem.feat import RdkitGridFeaturizer

def convert_with_obabel(input_path, output_path):
    try:
        subprocess.run(['obabel', input_path, '-O', output_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"Converted: {input_path} -> {output_path}")
    except subprocess.CalledProcessError:
        logging.warning(f"Open Babel conversion failed for: {input_path} -> {output_path}")

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

    required_columns = ['ligand', 'id', 'Resultado Experimental', 'Resultado Vina']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {', '.join(missing_columns)}")
        return

    # -------------------- Verificação e geração de arquivos obrigatórios -------------------- #
    logging.info("Verificando e gerando arquivos obrigatórios (SDF, PDB, PDBQT)...")

    for index, row in df.iterrows():
        complex_id = str(row['id'])
        complex_dir = os.path.join(args.output, complex_id)

        receptor_base = os.path.join(complex_dir, "receptor")
        ligand_base = os.path.join(complex_dir, "ligand")

        formats = ['sdf', 'pdb', 'pdbqt']

        for fmt in formats:
            receptor_file = f"{receptor_base}.{fmt}"
            if not os.path.exists(receptor_file):
                source = f"{receptor_base}.pdb" if fmt != 'pdb' else f"{receptor_base}.sdf"
                if os.path.exists(source):
                    convert_with_obabel(source, receptor_file)
                else:
                    logging.warning(f"Receptor source missing for {complex_id}: no {source}")

            ligand_file = f"{ligand_base}.{fmt}"
            if not os.path.exists(ligand_file):
                source = f"{ligand_base}.pdb" if fmt != 'pdb' else f"{ligand_base}.sdf"
                if os.path.exists(source):
                    convert_with_obabel(source, ligand_file)
                else:
                    logging.warning(f"Ligand source missing for {complex_id}: no {source}")

    # -------------------- Geração de SMILES -------------------- #
    smiles_list = []
    for index, row in df.iterrows():
        ligand_path = row['ligand']
        complex_id = row['id']
        complex_dir = os.path.join(args.output, complex_id)
        ligand_pdb = os.path.join(complex_dir, "ligand.pdb")

        if os.path.exists(ligand_pdb):
            mol = Chem.MolFromPDBFile(ligand_pdb, sanitize=False)
            if mol:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    smiles_list.append(smiles)
                except Exception as e:
                    logging.warning(f"Failed to sanitize/generate SMILES for complex {complex_id}: {e}")
                    smiles_list.append(None)
            else:
                logging.warning(f"Failed to generate SMILES for complex {complex_id}")
                smiles_list.append(None)
        else:
            logging.warning(f"Ligand file not found for complex {complex_id}")
            smiles_list.append(None)

    df['SMILES'] = smiles_list
    df.dropna(subset=['SMILES'], inplace=True)

    # -------------------- Morgan Fingerprints -------------------- #
    logging.info("Calculando Morgan fingerprints...")

    bit_vectors = []
    for smi in df['SMILES']:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            bit_vectors.append(arr)
        else:
            bit_vectors.append(np.zeros((2048,), dtype=int))

    morgan_df = pd.DataFrame(bit_vectors, columns=[f"bit_{i}" for i in range(2048)])
    df.reset_index(drop=True, inplace=True)
    morgan_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, morgan_df], axis=1)

    # -------------------- Extração de Binding Features -------------------- #
    logging.info("Extraindo features de ligação (Grid)...")
    logging.basicConfig(level=logging.ERROR)  # Suprime logs problemáticos do DeepChem

    grid_featurizer = RdkitGridFeaturizer(voxel_width=16.0, ecfp_power=5, splif_power=5, flatten=True)
    features_list = []

    for index, row in df.iterrows():
        dir_id = str(row['id'])  # Convertendo para string
        complex_dir = os.path.join(args.output, dir_id)
        receptor_pdb = os.path.join(complex_dir, "receptor.pdb")
        ligand_pdb = os.path.join(complex_dir, "ligand.pdb")

        try:
            features = grid_featurizer.featurize([(ligand_pdb, receptor_pdb)])
            features_list.append(features[0])
        except Exception as e:
            logging.warning(f"Failed to featurize datapoint {dir_id}. Appending empty array. Error: {e}")
            features_list.append([0] * 32)

    feature_columns = [f"GridFeature_{i}" for i in range(32)]
    expanded_features = pd.DataFrame(features_list, columns=feature_columns)
    df = pd.concat([df, expanded_features], axis=1)
    df.dropna(inplace=True)

    output_csv = os.path.join(args.output, 'processed_data_with_features.csv')
    df.to_csv(output_csv, index=False)
    logging.info(f"Processed data saved to {output_csv}")

if __name__ == "__main__":
    main()
