import os
import pandas as pd

# Carrega os dados do arquivo CSV, pré-processa e retorna X, y e os nomes das colunas
# Uso: X, y, feature_names = load_spaceship_data("data/spaceship_data.csv")
# X: matriz de features (numpy array)
# y: vetor de rótulos (numpy array)
# feature_names: lista de nomes das colunas (features)

def load_spaceship_data(path="data/spaceship_data.csv"):
    # Se o caminho não for absoluto e o ficheiro não existir, tente localizar
    # o ficheiro relativo à raiz do projecto (diretório pai de src/).
    if not os.path.isabs(path) and not os.path.exists(path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        alt_path = os.path.join(project_root, path)
        if os.path.exists(alt_path):
            path = alt_path

    df = pd.read_csv(path)

    y = df["Transported"].astype(int)  # True/False -> 1/0
    X = df.drop(columns=["Transported", "PassengerId", "Name"], errors='ignore')

    # Separa colunas numéricas e categóricas
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns
    num_cols = X.select_dtypes(include=["number"]).columns

    # Preenchimento simples
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

    # One-hot encoding
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X.values, y.values, list(X.columns)

