import os
import pandas as pd
import glob
import deepchem as dc

def extract_binding_features(input_dir):
    """
    Extrai features de binding usando RdkitGridFeaturizer
    
    Parâmetros:
    input_dir (str): Diretório contendo os complexos
    
    Retorna:
    pandas.DataFrame: DataFrame com features de binding
    """
    # Configuração do Grid Featurizer
    grid_featurizer = dc.feat.RdkitGridFeaturizer(
        voxel_width=16.0,  # Tamanho do voxel em Angstroms
        ecfp_power=5,      # Complexidade do descritor ECFP
        splif_power=5,     # Complexidade do descritor SPLIF
        flatten=True       # Achata o array de features
    )
    
    # Listas para armazenar dados
    data = []
    ids = []
    
    # Percorre todos os diretórios de complexos
    for directory in glob.glob(os.path.join(input_dir, '*')):
        # Verifica se é um diretório
        if not os.path.isdir(directory):
            continue
        
        # Extrai ID do complexo
        dir_id = os.path.basename(directory)
        
        # Caminhos para receptor e ligante
        receptor_pdb = os.path.join(directory, 'receptor.pdb')
        ligand_pdb = os.path.join(directory, 'ligand.pdb')
        
        # Verifica se os arquivos existem
        if not (os.path.exists(receptor_pdb) and os.path.exists(ligand_pdb)):
            print(f"Arquivos não encontrados para o complexo {dir_id}")
            continue
        
        try:
            # Gera features usando o GridFeaturizer
            features = grid_featurizer.featurize([(receptor_pdb, ligand_pdb)])
            
            # Adiciona features e ID
            data.append(features[0])
            ids.append(dir_id)
        
        except Exception as e:
            print(f"Erro ao processar complexo {dir_id}: {e}")
    
    # Cria DataFrame com features
    df_features = pd.DataFrame(data)
    df_features['ID'] = ids
    
    return df_features

def main():
    # Diretório de entrada dos complexos
    input_dir = '/home/lucasmocellin/tcc/melimda/Data/Complexos'
    
    # Diretório de saída para o CSV
    output_dir = '/home/lucasmocellin/tcc/melimda/Data/Complexos'
    output_csv = os.path.join(output_dir, 'binding_features.csv')
    
    # Extrai features
    df_features = extract_binding_features(input_dir)
    
    # Salva CSV
    df_features.to_csv(output_csv, index=False)
    
    print(f"Features de binding salvas em: {output_csv}")
    print(f"Total de complexos processados: {len(df_features)}")
    print(f"Número de features por complexo: {df_features.shape[1] - 1}")  # Subtrai 1 pela coluna de ID

if __name__ == "__main__":
    main()