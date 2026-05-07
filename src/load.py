import os
import pandas as pd


def _resolve_data_path(path: str) -> str:
    if not os.path.isabs(path) and not os.path.exists(path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        alt_path = os.path.join(project_root, path)
        if os.path.exists(alt_path):
            path = alt_path
    return path

# Carrega os dados do arquivo CSV, pré-processa e retorna X, y e os nomes das colunas
# Uso: X, y, feature_names = load_spaceship_data("data/spaceship_data.csv")
# X: matriz de features (numpy array)
# y: vetor de rótulos (numpy array)
# feature_names: lista de nomes das colunas (features)

def load_spaceship_data(path="data/spaceship_data.csv"):
    df = load_spaceship_data_as_df(path)

    y = df["Transported"].astype(int)  # True/False -> 1/0
    X = df.drop(columns=["Transported"], errors='ignore')

    # Keep the original 13 feature columns as-is for analysis/feature inspection.
    # This avoids exploding Cabin into thousands of dummy columns.
    if "PassengerId" in X.columns:
        X["PassengerId"] = X["PassengerId"].fillna(X["PassengerId"].mode().iloc[0])
    if "Name" in X.columns:
        X["Name"] = X["Name"].fillna(X["Name"].mode().iloc[0])

    return X.values, y.values, list(X.columns)

def load_spaceship_data_as_df(path="data/spaceship_data.csv"):
    path = _resolve_data_path(path)
    df = pd.read_csv(path)
    return df
