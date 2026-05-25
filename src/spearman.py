"""Spearman correlation analysis utilities."""
#author: Jaime Sousa

import pandas as pd
from scipy import stats
import importlib
from src import load as load_module


def spearman_with_target(df, target_col, filter_col=None, filter_value=None):
    """
    Compute Spearman correlations between features and target column.
    Optionally filter the data by a column condition (e.g., Transported == True).
    
    Parameters:
    -----------
    path : str
        Path to the data CSV file.
    target_col : str
        Name of the target column.
    filter_col : str, optional
        Column name to filter on (e.g., 'Transported' for O3).
    filter_value : any, optional
        Value to filter for in filter_col (e.g., True for O3).
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with correlations and p-values, sorted by absolute correlation.
    """
    # importlib.reload(load_module)

    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col].astype(int)
    feature_names = list(df.columns)
    
    # X, y, feature_names = load_module.load_spaceship_data(path, target_col=target_col)
    X_df = pd.DataFrame(X, columns=feature_names).copy()
    y_s = pd.Series(y, name=target_col).copy()

    # If filter is specified, add the filter column and apply it.
    # Made for O3 where we want to filter to only transported passengers.
    if filter_col is not None and filter_value is not None:
        # raw_df = load_module.load_spaceship_data_as_df(path)
        filter_mask = df[filter_col] == filter_value
        X_df = X_df.loc[filter_mask].reset_index(drop=True)
        y_s = y_s.loc[filter_mask].reset_index(drop=True)

    # Fill missing values for feature columns
    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
            X_df[col] = X_df[col].fillna(X_df[col].median())
        else:
            X_df[col] = X_df[col].fillna(X_df[col].mode().iloc[0])

    # Ensure numeric target for Spearman
    y_s = pd.to_numeric(y_s, errors="coerce")
    y_s = y_s.fillna(y_s.median())

    correlations = {}
    for col in X_df.columns:
        series = X_df[col]
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.Categorical(series).codes

        corr, p_value = stats.spearmanr(series, y_s)
        correlations[col] = {"correlation": corr, "p_value": p_value}

    return pd.DataFrame(correlations).T.sort_values("correlation", key=abs, ascending=False)
