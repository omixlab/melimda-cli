import pandas as pd
import os

# Caminhos
tabela_principal_path = '/home/lucasmocellin/tcc/melimda/predicted_results_with_ids_refined.csv'  # Substitua pelo caminho real
pasta_melimda = '/home/lucasmocellin/tcc/melimda/Data/melimda'

# 1. Ler a tabela principal
df_principal = pd.read_csv(tabela_principal_path)

# 2. Remover a coluna "Real"
df_principal = df_principal.drop(columns=['Real'])

# 3. Lista para armazenar os novos dados
dados_unificados = []

# 4. Para cada ID, buscar o resumo_poses.csv correspondente
for _, row in df_principal.iterrows():
    complexo_id = row['ID']
    predicted_geral = row['Predicted']
    
    resumo_path = os.path.join(pasta_melimda, complexo_id, 'resumo_poses.csv')
    
    if os.path.exists(resumo_path):
        df_resumo = pd.read_csv(resumo_path)
        
        for _, resumo_row in df_resumo.iterrows():
            dados_unificados.append({
                'ID': complexo_id,
                'Predicted_geral': predicted_geral,
                'pose_id': resumo_row['pose_id'],
                'vina_result': resumo_row['vina_result'],
                'Prediction_pose': resumo_row['Prediction']
            })
    else:
        print(f'[AVISO] Arquivo não encontrado: {resumo_path}')

# 5. Criar DataFrame final e salvar
df_final = pd.DataFrame(dados_unificados)
df_final.to_csv('tabela_unificada.csv', index=False)
