import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
def load_data(filepath):
    # loading dataset
    df = pd.read_csv(filepath)
    return df

if __name__ == "__main__":
    data = load_data('C:/Users/USER/Documents/Master2 DIT/Outil versioning/exam_versioning/CarPrice.csv')
    print (data.head())