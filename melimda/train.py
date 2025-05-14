import argparse
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

def train_model(file_path, model_output):
    # Carregar o dataset
    df = pd.read_csv(file_path)

    # Definir colunas
    colunas_experimento = ["Resultado Experimental"]  # Energia experimental (ground truth)
    colunas_vina = ["Resultado Vina"]  # Energia prevista pelo Vina
    colunas_morgan = [f"bit_{i}" for i in range(2048)]  # Morgan Fingerprints
    colunas_grid = [f"GridFeature_{i}" for i in range(32)]  # 32 colunas do Grid Featurizer

    # Garantir que todas as colunas existam no dataset
    colunas_necessarias = colunas_experimento + colunas_vina + colunas_morgan + colunas_grid
    for col in colunas_necessarias:
        if col not in df.columns:
            raise ValueError(f"Coluna faltando no dataset: {col}")

    # Definir features e target
    X = df[colunas_vina + colunas_morgan + colunas_grid]  # Substituído: MACAW → Morgan
    Y = df["Resultado Experimental"]

    # Separar os dados
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Criar o modelo CatBoost
    modelo = CatBoostRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        depth=10,
        loss_function='MAE',
        verbose=100
    )

    # Criar Pool de dados
    train_pool = Pool(X_train, Y_train)
    test_pool = Pool(X_test, Y_test)

    # Treinar o modelo
    
    modelo.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)
    modelo.fit(X_train, Y_train)
    # Previsões
    Y_pred = modelo.predict(X_test)

    # Avaliações
    mape = mean_absolute_percentage_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print(f"MAPE: {mape}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print(f"Raiz de R² (r): {r2 ** 0.5}")

    # Salvar o modelo
    modelo.save_model(model_output)

def main():
    parser = argparse.ArgumentParser(description="Treina um modelo CatBoost para re-scoring de docking molecular.")
    parser.add_argument("--file_path", type=str, required=True, help="Caminho para o arquivo CSV com os dados.")
    parser.add_argument("--model_output", type=str, default="catboost_redocking.cbm", help="Caminho para salvar o modelo treinado.")
    args = parser.parse_args()
    train_model(file_path=args.file_path, model_output=args.model_output)

