#/backend/prediction.py
import os
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from data_processing import DataProcessor
from utils import inverse_transform

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model_dir, data_dir, time_step):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.time_step = time_step

    def get_model_path(self, symbol, model_type):
        if model_type in ['LSTM', 'TCN', 'Transformer']:
            model_extension = 'keras'
        else:
            model_extension = 'pkl'
        model_filename = f'{symbol}_{model_type}_model.{model_extension}'
        return os.path.join(self.model_dir, model_filename)

    def load_model(self, model_type, model_path):
        if model_type in ['LSTM', 'TCN', 'Transformer']:
            return load_model(model_path, compile=False)
        else:
            return joblib.load(model_path)

    def predict_model(self, model, model_type, X):
        if model_type in ['LSTM', 'TCN', 'Transformer']:
            y_pred = model.predict(X)
        else:
            X_flat = X.reshape(X.shape[0], -1)
            y_pred = model.predict(X_flat)
            y_pred = y_pred.reshape(-1, 1)
        return y_pred

    def predict(self, symbol, model_type='LSTM', days_ahead=1):
        model_path = self.get_model_path(symbol, model_type)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Mô hình {model_type} cho {symbol} không tồn tại.")

        model = self.load_model(model_type, model_path)

        data_processor = DataProcessor(self.data_dir, symbol)
        # Load và xử lý dữ liệu
        data = data_processor.load_data()
        if data is None:
            raise ValueError(f"Dữ liệu cho {symbol} không tồn tại.")
        # Thêm các chỉ báo kỹ thuật
        from feature_engineering import add_technical_indicators
        data = add_technical_indicators(data.reset_index())

        scaled_data, scaler = data_processor.preprocess_data(data)
        data_processor.scaler = scaler
        # Xác định vị trí của target, ví dụ 'Close'
        target_index = data_processor.feature_columns.index('Close')
        # Lấy ra chuỗi cuối cùng
        last_data = scaled_data[-self.time_step:]
        predictions = []
        for _ in range(days_ahead):
            input_data = last_data.reshape(1, self.time_step, -1)
            y_pred_scaled = self.predict_model(model, model_type, input_data)
            y_pred = inverse_transform(scaler, y_pred_scaled, target_index)
            predictions.append(y_pred[0])
            new_input = np.append(last_data[1:], y_pred_scaled.reshape(1, -1), axis=0)
            last_data = new_input
        return predictions

    def visualize_predictions(self, symbol, model_type='LSTM'):
        """Enhanced visualization with confidence intervals and multiple metrics"""
        import seaborn as sns
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
        # Load and prepare data
        model = self.load_model(model_type, self.get_model_path(symbol, model_type))
        data_processor = DataProcessor(self.data_dir, symbol)
        data = data_processor.load_data()
    
        if data is None:
            raise ValueError(f"Data for {symbol} not found")
        
        data = add_technical_indicators(data.reset_index())
        scaled_data, scaler = data_processor.preprocess_data(data)
        target_index = data_processor.feature_columns.index('Close')
    
        # Create sequences and predictions
        X, y = data_processor.create_sequences(scaled_data, self.time_step, target_index)
        split_index = int(len(X) * 0.8)
        X_test, y_test = X[split_index:], y[split_index:]
    
        # Generate predictions with confidence intervals
        y_pred_scaled = self.predict_model(model, model_type, X_test)
        y_test_inv = inverse_transform(scaler, y_test.reshape(-1, 1), target_index)
        y_pred_inv = inverse_transform(scaler, y_pred_scaled, target_index)
    
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
    
        # Create visualization
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
        # Price plot with confidence intervals
        ax1.plot(y_test_inv, label='Actual', color='blue', alpha=0.7)
        ax1.plot(y_pred_inv, label='Predicted', color='red', alpha=0.7)
    
        # Add confidence intervals
        std_dev = np.std(y_test_inv - y_pred_inv)
        ax1.fill_between(range(len(y_pred_inv)),
                        y_pred_inv - 2*std_dev,
                        y_pred_inv + 2*std_dev,
                        color='red', alpha=0.1,
                        label='95% Confidence Interval')
    
        ax1.set_title(f'{symbol} Stock Price Prediction using {model_type}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
    
        # Error distribution plot
        errors = y_test_inv - y_pred_inv
        sns.histplot(errors, kde=True, ax=ax2)
        ax2.set_title('Prediction Error Distribution')
        ax2.set_xlabel('Error')
        ax2.set_ylabel('Frequency')
    
        # Add metrics text
        metrics_text = f'RMSE: {np.sqrt(mse):.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}'
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
        plt.tight_layout()
    
        # Save the plot
        plot_path = os.path.join(self.model_dir, f'{symbol}_{model_type}_prediction_analysis.png')
        plt.savefig(plot_path)
        plt.close()
    
        return {
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'plot_path': plot_path
        }
