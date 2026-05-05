import pandas as pd

# Carrega os dados do arquivo CSV, pré-processa e retorna X, y e os nomes das colunas
# Uso: X, y, feature_names = load_spaceship_data("data/spaceship_data.csv")
# X: matriz de features (numpy array)
# y: vetor de rótulos (numpy array)
# feature_names: lista de nomes das colunas (features)

def load_spaceship_data(path="data/spaceship_data.csv"):
    df = pd.read_csv(path)

    y = df["Transported"].astype(int)  # True/False -> 1/0
    X = df.drop(columns=["Transported", "PassengerId", "Name"])

    # Separa colunas numéricas e categóricas
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns
    num_cols = X.select_dtypes(include=["number"]).columns

    # Preenchimento simples
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

    # One-hot encoding
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X.values, y.values, X.columns

