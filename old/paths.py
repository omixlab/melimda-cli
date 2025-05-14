import os
import shutil
import pandas as pd

def copy_folders(csv_path, destination):
    # Carregar o CSV
    df = pd.read_csv(csv_path)
    
    # Criar o diretório de destino se não existir
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    # Iterar sobre as linhas do CSV
    for _, row in df.iterrows():
        complex_path = os.path.dirname(row['ligand'])  # Assumindo que ligante e receptor estão na mesma pasta
        complex_id = row['id']
        
        # Definir diretório de destino
        complex_dest = os.path.join(destination, complex_id)
        
        # Copiar a pasta do complexo
        if os.path.exists(complex_path):
            shutil.copytree(complex_path, complex_dest, dirs_exist_ok=True)
            print(f"Copiado: {complex_path} -> {complex_dest}")
        else:
            print(f"Aviso: Pasta do complexo não encontrada - {complex_path}")

if __name__ == "__main__":
    csv_file = "/home/lucasmocellin/tcc/melimda/Data/caminhos_resultados.tiny.csv"  # Atualize com o caminho real
    destination_folder = "/home/lucasmocellin/tcc/melimda/Data/Complexos"  # Atualize com o destino desejado
    copy_folders(csv_file, destination_folder)