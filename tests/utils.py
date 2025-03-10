#/utils.py
import numpy as np
from scipy import stats 

def inverse_transform(scaler, y_data, target_index):
    # Tạo mảng zero có cùng số lượng feature
    zeros = np.zeros((len(y_data), scaler.mean_.shape[0]))
    zeros[:, target_index] = y_data.flatten()
    y_inv = scaler.inverse_transform(zeros)[:, target_index]
    return y_inv

def SMAPE(y_true, y_pred):
    epsilon = 1e-10  # Để tránh chia cho 0
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return 100 * np.mean(numerator / denominator)

def evaluate_model_performance(y_true, y_pred, model_name):
    """Comprehensive model evaluation metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {
        'model_name': model_name,
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'smape': SMAPE(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'direction_accuracy': np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
    }
    
    # Calculate additional statistics
    errors = y_true - y_pred
    results.update({
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'skewness': stats.skew(errors),
        'kurtosis': stats.kurtosis(errors)
    })
    
    return results