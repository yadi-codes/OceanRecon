import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Features and target definitions
FEATURE_COLS = ['latitude', 'longitude', 'sst', 'ssh', 'sss', 'ussw', 'vssw', 'depth']
TARGET_COL = 'temp'

def get_model_instance(model_name, hyperparams=None):
    """
    Returns an instance of the requested model with specified hyperparameters.
    """
    if hyperparams is None:
        hyperparams = {}
        
    if model_name == "Linear Regression":
        return LinearRegression()
        
    elif model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', 10),
            min_samples_split=hyperparams.get('min_samples_split', 2),
            random_state=hyperparams.get('random_state', 42),
            n_jobs=-1
        )
        
    elif model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', 6),
            learning_rate=hyperparams.get('learning_rate', 0.1),
            random_state=hyperparams.get('random_state', 42),
            n_jobs=-1
        )
        
    elif model_name == "LightGBM":
        return LGBMRegressor(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', -1),
            learning_rate=hyperparams.get('learning_rate', 0.1),
            random_state=hyperparams.get('random_state', 42),
            n_jobs=-1,
            verbose=-1
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_and_evaluate(df, model_name, hyperparams=None, train_size=0.8, random_state=42):
    """
    Splits the data, fits a standard scaler, trains the model, and calculates metrics.
    
    Returns:
    --------
    dict containing:
        - 'model': trained model object
        - 'scaler': fitted StandardScaler object
        - 'train_time': elapsed time in seconds
        - 'metrics_train': dict of overall train metrics
        - 'metrics_test': dict of overall test metrics
        - 'y_train_pred': np.array
        - 'y_test_pred': np.array
        - 'df_test_eval': pd.DataFrame with truth vs pred and features
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)
    
    # Get model and train
    model = get_model_instance(model_name, hyperparams)
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics_train = calculate_metrics(y_train, y_train_pred)
    metrics_test = calculate_metrics(y_test, y_test_pred)
    
    # Combine test features, truth and predictions for plotting
    df_test_eval = X_test.copy()
    df_test_eval['temp_true'] = y_test.values
    df_test_eval['temp_pred'] = y_test_pred
    df_test_eval['error'] = df_test_eval['temp_pred'] - df_test_eval['temp_true']
    df_test_eval['abs_error'] = df_test_eval['error'].abs()
    
    return {
        'model': model,
        'scaler': scaler,
        'train_time': train_time,
        'metrics_train': metrics_train,
        'metrics_test': metrics_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'df_test_eval': df_test_eval
    }

def calculate_metrics(y_true, y_pred):
    """
    Computes overall regression metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }

def get_depth_wise_metrics(df_eval):
    """
    Computes metrics grouped by depth levels.
    
    Parameters:
    -----------
    df_eval : pd.DataFrame
        DataFrame with columns 'depth', 'temp_true', 'temp_pred'
        
    Returns:
    --------
    pd.DataFrame
        Grouped metrics with columns: ['depth', 'rmse', 'mae', 'r2', 'count']
    """
    results = []
    grouped = df_eval.groupby('depth')
    
    for depth, group in grouped:
        y_true = group['temp_true']
        y_pred = group['temp_pred']
        
        # Guard against single sample edge case
        if len(y_true) < 2:
            r2 = 0.0
        else:
            r2 = r2_score(y_true, y_pred)
            
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        results.append({
            'depth': float(depth),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'count': int(len(group))
        })
        
    return pd.DataFrame(results).sort_values('depth')
