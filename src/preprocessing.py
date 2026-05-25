"""Utils for doing data preprocessing such as scaling and imputation"""
#Author: Juniper Wilson 61795
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""Standardizes the data of a DataFrame of the training data, 
    returns a new DataFrame with the numerical columns all standardized
"""
def standardize(numerical_cols: list[str], categorical_cols: list[str], numerical_df: pd.DataFrame, categorical_df: pd.DataFrame) -> pd.DataFrame:
    # Scaling The Data
    # data scaling is needed for the regression algorithms to work properly
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(numerical_df)
    numerical_df = pd.DataFrame(scaled_df, columns = numerical_cols)

    return categorical_df.join(numerical_df)
    
"""Imputes the missing values of the data using the KNNImputer for numerical columns and SimpleImputer for categorical columns,
    returns a DataFrame that is the joined numerical_df and categorical_df with imputed values
"""
def impute(numerical_cols: list[str], categorical_cols: list[str], numerical_df: pd.DataFrame, categorical_df: pd.DataFrame) -> pd.DataFrame:
    # Imputing The Data
    # knnimputer can be used for numerical values, something else should be used for categorical values, 
    ki = KNNImputer(
        missing_values=np.nan,
        n_neighbors=5,
        copy=True
                    )
    numerical_df = ki.fit_transform(numerical_df)
    numerical_df = pd.DataFrame(numerical_df, columns = numerical_cols)

    # simple imputer is used for categorical values
    si = SimpleImputer(
        missing_values=np.nan,
        strategy="most_frequent"
    )
    si.fit(categorical_df)
    categorical_df = si.fit_transform(categorical_df)
    categorical_df = pd.DataFrame(categorical_df, columns = categorical_cols)

    return categorical_df.join(numerical_df)





        

