import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_processdata(path):
    data = pd.read_csv(path)
    
    x = data.drop('MEDV' , axis=1)
    y = data['MEDV']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    return x_scaled , y , scaler

def save_preprocessing(scaler, feature_names, scaler_path, feature_names_path):
    import joblib
    
    joblib.dump(scaler, scaler_path)
    
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_names))