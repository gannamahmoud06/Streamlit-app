import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_and_preprocessdata, save_preprocessing

def train_house_price_model(dataset_path, model_path, scaler_path, feature_names_path):

    x_scaled, y, scaler, feature_names = load_and_preprocessdata(dataset_path)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        max_depth=10
    )
    rf_model.fit(x_train, y_train)
    
    y_pred = rf_model.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    joblib.dump(rf_model, model_path)
    
    save_preprocessing(
        scaler, feature_names, 
        scaler_path, feature_names_path
    )
    
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importances for House Price Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'R-squared': r2
    }

if __name__ == '__main__':
    metrics = train_house_price_model(
        dataset_path='dataset\boston.csv',
        model_path='trained_model.pkl',
        scaler_path='scaler.pkl',
        feature_names_path='feature_names.txt'
    )
    print("Model Training Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")