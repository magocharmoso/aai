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

def load_spaceship_data(path="data/spaceship_data.csv", target_col="Transported"):
    df = load_spaceship_data_as_df(path)

    # Ensure the target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Extract target (attempt sensible conversions: bool->int, 'True'/'False' strings, numeric)
    s = df[target_col]
    if pd.api.types.is_bool_dtype(s):
        y = s.astype(int).values
    else:
        # handle string booleans
        uniq = pd.Series(s.dropna().unique()).astype(str).str.lower().unique()
        if set(uniq).issubset({"true", "false"}):
            y = s.astype(str).str.lower().map({"true": 1, "false": 0}).values
        else:
            # try numeric conversion; if fails, return raw values
            y_num = pd.to_numeric(s, errors='coerce')
            if y_num.notna().all():
                y = y_num.values
            else:
                y = s.values

    X = df.drop(columns=[target_col], errors='ignore')

    # Keep the original 13 feature columns as-is for analysis/feature inspection.
    # This avoids exploding Cabin into thousands of dummy columns.
    if "PassengerId" in X.columns:
        X["PassengerId"] = X["PassengerId"].fillna(X["PassengerId"].mode().iloc[0])
    if "Name" in X.columns:
        X["Name"] = X["Name"].fillna(X["Name"].mode().iloc[0])

    # Ensure we always return numpy arrays for X and y
    y_out = y.values if hasattr(y, "values") else pd.Series(y).values
    return X.values, y_out, list(X.columns)

def load_spaceship_data_as_df(path="data/spaceship_data.csv"):
    path = _resolve_data_path(path)
    df = pd.read_csv(path)
    return df
