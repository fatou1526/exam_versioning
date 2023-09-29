import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
def load_data(filepath):
    # loading dataset
    df = pd.read_csv(filepath)
    return df

def preprocessing_duplicates(data_df):
    # This method help to drop duplicates
    if data_df.duplicated().sum() == 0:
        print("No duplicates")
        data_df_nd = data_df
    else:
        print("Remove duplicates")
        data_df_nd = data_df.drop_duplicates()
    return data_df_nd

def preprocessing_null(data_df):
    # This method help to drop NaN values
    data_nn = data_df.dropna()
    return data_nn

def preprocessing_outliers(data_df):
    # Here, the method helps to remove outliers
    col = data_df.columns
    for i in col:
        col_skew = data_df[i].skew()
        print(f"The skew for column {i} is {col_skew}")
        if (col_skew < -1) | (col_skew > 1):
            print(f"Column {i} has outliers")
            data_df[i] = np.where(data_df[i] < data_df[i].quantile(0.1), data_df[i].quantile(0.1), data_df[i])
            data_df[i] = np.where(data_df[i] > data_df[i].quantile(0.9), data_df[i].quantile(0.9), data_df[i])
            print(f" The column {i} has been processed and the skew for column {i} is now {data_df[i].skew()}")
        else:
            print(f"There are not outliers at column {i}")
    return data_df

if __name__ == "__main__":
    data = load_data('CarPrice.csv')
    print (data.head())

    # Remove all categorical columns
    data_num = data.select_dtypes(exclude=object)

    # Preprocessing data (duplicates, outliers, null )
    data = preprocessing_duplicates(data_num)
    data = preprocessing_outliers(data_num)
    data = preprocessing_null(data_num)