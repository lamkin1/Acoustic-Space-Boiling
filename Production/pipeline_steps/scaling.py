import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess(data):
    string_cols = []
    na_cols = []
    numeric_cols = []
    for col in data.columns:
        if pd.api.types.is_string_dtype(data[col]):
            string_cols.append(col)
        elif data[col].isnull().any():
            na_cols.append(col)
        else:
            numeric_cols.append(col)
    removed_cols = string_cols + na_cols
    if removed_cols:
        print("Removed columns:", removed_cols)
        data = data[numeric_cols]
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(data)
    return scaledData, numeric_cols
