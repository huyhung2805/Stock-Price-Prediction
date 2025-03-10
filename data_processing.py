import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_dir, symbol):
        self.data_dir = data_dir
        self.symbol = symbol
        self.feature_columns = None
        self.scaler = None

    def load_data(self):
        file_path = os.path.join(self.data_dir, f'{self.symbol}_historical_data.csv')
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist.")
            return None

        data = pd.read_csv(file_path)
        data.rename(columns=lambda x: x.strip(), inplace=True)

        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in expected_columns):
            logger.warning(f"Data in {file_path} missing columns or column names don't match.")
            return None

        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values('Date', inplace=True)
        data = data[expected_columns[1:]]

        self.feature_columns = data.columns.tolist()
    
        # Handle missing values - updated method
        data = self.handle_missing_values(data)
    
        # Remove outliers
        data = self.remove_outliers(data)
    
        return data

    def handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        # First, interpolate missing values
        data = data.interpolate(method='linear')
    
        # Then fill any remaining NaNs at the edges
        data = data.ffill()  # Forward fill
        data = data.bfill()  # Backward fill
    
        return data

    def remove_outliers(self, data):
        z_scores = np.abs(stats.zscore(data[self.feature_columns]))
        filtered_entries = (z_scores < 3).all(axis=1)
        data = data[filtered_entries]
        return data

    def preprocess_data(self, data, scaler=None):
        """
        Preprocess data: normalize features
        """
        if isinstance(data, pd.DataFrame):
            data_values = data.values
        else:
            data_values = data
            
        if scaler is None:
            scaler = RobustScaler()  # Using RobustScaler for better outlier handling
            scaled_data = scaler.fit_transform(data_values)
        else:
            scaled_data = scaler.transform(data_values)
            
        self.scaler = scaler
        logger.debug(f"Scaled data shape: {scaled_data.shape}")
        return scaled_data, scaler

    def create_sequences(self, scaled_data, time_step, target_column):
        """
        Create sequences of data for time series prediction.
        """
        try:
            X, y = [], []
            
            if len(scaled_data) <= time_step:
                logger.warning(f"Not enough data to create sequences. Data length: {len(scaled_data)}, Time step: {time_step}")
                return np.array(X), np.array(y)
            
            for i in range(len(scaled_data) - time_step):
                sequence = scaled_data[i:(i + time_step)]
                X.append(sequence)
                target = scaled_data[i + time_step, target_column]
                y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            logger.debug(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            raise
