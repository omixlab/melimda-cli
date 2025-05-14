import os
import argparse
import logging
import subprocess
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor, Pool
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from deepchem.feat import RdkitGridFeaturizer


def extract_vina_result(pdbqt_path):
    """Extrai o resultado do Vina de um arquivo PDBQT."""
    with open(pdbqt_path, 'r') as file:
        for line in file:
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.strip().split()
                if len(parts) >= 4:
                    return float(parts[3])
    raise ValueError("VINA RESULT não encontrado no arquivo.")


def convert_with_obabel(input_path, output_path):
    """Converte entre formatos usando Open Babel."""
    try:
        subprocess.run(['obabel', input_path, '-O', output_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"Convertido: {input_path} -> {output_path}")
        return True
    except subprocess.CalledProcessError:
        logging.warning(f"Falha ao converter com Open Babel: {input_path} -> {output_path}")
        return False


def split_vina_poses(ligand_pdbqt, output_dir):
    """
    Usa o comando vina_split para separar as diferentes poses no arquivo PDBQT.
    Retorna uma lista com os caminhos para cada arquivo de pose.
    """
    os.makedirs(output_dir, exist_ok=True)
    split_dir = os.path.join(output_dir, "split_poses")
    os.makedirs(split_dir, exist_ok=True)
    
    # Nome base para as poses separadas
    base_name = os.path.join(split_dir, "pose")
    
    try:
        cmd = ['vina_split', '--input', ligand_pdbqt, '--ligand', base_name]
        logging.info(f"Executando comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.debug(f"Saída do vina_split: {result.stdout}")
        
        # Listar todos os arquivos gerados
        pose_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.startswith("pose") and f.endswith(".pdbqt")]
        pose_files.sort()  # Garantir que estejam em ordem
        
        logging.info(f"Separadas {len(pose_files)} poses do arquivo PDBQT")
        return pose_files
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar vina_split: {e}")
        logging.error(f"Stderr: {e.stderr}")
        return []
    except Exception as e:
        logging.error(f"Erro ao separar poses: {e}")
        return []


def preprocess_single_pose(ligand_pdbqt, receptor_pdb, output_dir, pose_id):
    """
    Pré-processa uma única pose do ligante e extrai características.
    """
    pose_dir = os.path.join(output_dir, f"pose_{pose_id}")
    os.makedirs(pose_dir, exist_ok=True)
    complex_id = f"complex_pose_{pose_id}"

    # Caminhos dos arquivos para esta pose
    ligand_pdb = os.path.join(pose_dir, f"ligand.pdb")
    receptor_pdb_out = os.path.join(pose_dir, "receptor.pdb")

    # Converte ligand e receptor para PDB
    if not convert_with_obabel(ligand_pdbqt, ligand_pdb):
        logging.error(f"Falha ao converter ligante para pose {pose_id}")
        return None

    if not os.path.exists(receptor_pdb_out):
        if not convert_with_obabel(receptor_pdb, receptor_pdb_out):
            logging.error(f"Falha ao converter receptor para pose {pose_id}")
            return None

    # Extrair valor do vina
    try:
        vina_result_value = extract_vina_result(ligand_pdbqt)
    except ValueError as e:
        logging.error(f"Erro ao extrair resultado do Vina para pose {pose_id}: {e}")
        return None

    # Criar DataFrame inicial
    df = pd.DataFrame([{
        'id': complex_id,
        'pose_id': pose_id,
        'ligand_path': ligand_pdbqt,
        'receptor_path': receptor_pdb,
        'vina_result': vina_result_value
    }])

    # Gerar SMILES
    try:
        mol = Chem.MolFromPDBFile(ligand_pdb, sanitize=False)
        if mol:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            df['SMILES'] = [smiles]
        else:
            logging.warning(f"Não foi possível gerar SMILES para pose {pose_id}")
            df['SMILES'] = [None]
            return None
    except Exception as e:
        logging.error(f"Erro ao gerar SMILES para pose {pose_id}: {e}")
        df['SMILES'] = [None]
        return None

    # Morgan fingerprints
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
        else:
            arr = np.zeros((2048,), dtype=int)
    except Exception as e:
        logging.error(f"Erro ao gerar fingerprints para pose {pose_id}: {e}")
        arr = np.zeros((2048,), dtype=int)

    for i in range(2048):
        df[f'bit_{i}'] = arr[i]

    # Grid features
    featurizer = RdkitGridFeaturizer(voxel_width=16.0, ecfp_power=5, splif_power=5, flatten=True)
    try:
        features = featurizer.featurize([(ligand_pdb, receptor_pdb_out)])[0]
    except Exception as e:
        logging.warning(f"Falha ao extrair grid features para pose {pose_id}: {e}")
        features = [0.0] * 32

    for i, val in enumerate(features):
        df[f'GridFeature_{i}'] = val

    return df


def predict_single_pose(df, model_path):
    """
    Faz predição para uma única pose usando o modelo.
    """
    if df is None or df.empty:
        return None

    model = CatBoostRegressor().load_model(model_path)

    feature_cols = [col for col in df.columns if col.startswith('bit_') or col.startswith('GridFeature_')]
    X = df[['vina_result', *feature_cols]]
    X = X.rename(columns={'vina_result': 'Resultado Vina'})
    prediction = model.predict(X)
    df['Prediction'] = prediction

    return df


def process_all_poses(ligand_pdbqt, receptor_pdb, model_path, output_dir):
    """
    Processa todas as poses do arquivo PDBQT.
    """
    # Separar as poses do arquivo PDBQT
    pose_files = split_vina_poses(ligand_pdbqt, output_dir)
    
    if not pose_files:
        logging.error("Não foi possível separar as poses. Verifique se o vina_split está instalado e no PATH.")
        return None

    # Processar cada pose separadamente
    all_results = []
    
    for i, pose_file in enumerate(pose_files):
        pose_id = str(i+1)
        logging.info(f"Processando pose {pose_id} - arquivo: {pose_file}")
        
        # Pré-processar a pose
        df = preprocess_single_pose(pose_file, receptor_pdb, output_dir, pose_id)
        
        if df is not None:
            # Fazer predição
            df = predict_single_pose(df, model_path)
            
            if df is not None:
                all_results.append(df)
                
                # Salvar resultado individual desta pose
                pose_results_dir = os.path.join(output_dir, f"pose_{pose_id}")
                pose_csv = os.path.join(pose_results_dir, f"resultado_pose_{pose_id}.csv")
                df.to_csv(pose_csv, index=False)
                logging.info(f"Resultado para pose {pose_id} salvo em: {pose_csv}")

    if all_results:
        # Juntar todos os resultados
        final_df = pd.concat(all_results, ignore_index=True)
        return final_df
    else:
        logging.error("Não foi possível processar nenhuma pose com sucesso.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Rodar predição com modelo treinado para múltiplas poses")
    parser.add_argument('--ligand', required=True, help="Arquivo .pdbqt de saída do Vina")
    parser.add_argument('--receptor', required=True, help="Arquivo .pdb do receptor")
    parser.add_argument('--model', required=True, help="Arquivo .pkl do modelo treinado")
    parser.add_argument('--output', required=True, help="Diretório para arquivos temporários e resultados")
    parser.add_argument('--debug', action='store_true', help="Ativar modo de depuração com logs detalhados")
    args = parser.parse_args()

    # Configurar logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(args.output, "process.log")),
                            logging.StreamHandler()
                        ])
    
    logging.info("Iniciando processamento de múltiplas poses")
    logging.info(f"Ligante: {args.ligand}")
    logging.info(f"Receptor: {args.receptor}")
    logging.info(f"Modelo: {args.model}")
    logging.info(f"Diretório de saída: {args.output}")
    
    # Criar diretório de saída
    os.makedirs(args.output, exist_ok=True)
    
    # Processar todas as poses
    final_df = process_all_poses(args.ligand, args.receptor, args.model, args.output)
    
    if final_df is not None:
        # Ordenar por predição (melhor primeiro)
        final_df = final_df.sort_values(by='Prediction', ascending=True)
        
        # Salvar resultados completos
        output_csv = os.path.join(args.output, 'resultado_final_todas_poses.csv')
        final_df.to_csv(output_csv, index=False)
        logging.info(f"Resultados consolidados salvos em: {output_csv}")
        
        # Salvar resumo
        summary_csv = os.path.join(args.output, 'resumo_poses.csv')
        final_df[['id', 'pose_id', 'vina_result', 'Prediction']].to_csv(summary_csv, index=False)
        logging.info(f"Resumo salvo em: {summary_csv}")
        
        # Exibir resultados resumidos
        print("\nResultados (ordenados pelo valor predito):")
        print(final_df[['id', 'pose_id', 'vina_result', 'Prediction']].to_string(index=False))
        
        # Análise das melhores poses
        print("\nAnálise das 3 melhores poses:")
        top3 = final_df.nsmallest(3, 'Prediction')
        for i, row in top3.iterrows():
            print(f"Pose {row['pose_id']} - Vina: {row['vina_result']:.2f} - Predição: {row['Prediction']:.2f}")
    else:
        logging.error("Não foi possível processar nenhuma pose.")


if __name__ == "__main__":
    main()