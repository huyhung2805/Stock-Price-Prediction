import os
import time
import logging
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (LSTM, GRU, Bidirectional, Dense, Dropout, 
                                   Input, LayerNormalization, Add, 
                                   MultiHeadAttention, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

import optuna
from optuna.integration import TFKerasPruningCallback
from tcn import TCN
from utils import SMAPE, evaluate_model_performance

logger = logging.getLogger(__name__)

# Thêm định nghĩa model_types
model_types = ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer', 'XGBoost', 'RF']

class ModelTrainer:
    def __init__(self, model_dir, time_step, num_features):
        self.model_dir = model_dir
        self.time_step = time_step
        self.num_features = num_features
        self.models = {}
        self.scaler = None
        self.power_transformer = None
        self.epochs = 200
        self.batch_size = 32
        self.n_trials = 20
        
        # Ensure model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def train_models(self, X_train, y_train, X_test, y_test, symbol):
        """Train multiple models and create ensemble predictions"""
        try:
            # Lưu trữ giá trị gốc trước khi transform
            self.y_scaler = MinMaxScaler()
            y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1))
            y_test_scaled = self.y_scaler.transform(y_test.reshape(-1, 1))

            all_predictions = {}
            all_metrics = []
            trained_models = {}

            for model_type in model_types:
                try:
                    logger.info(f"Training {model_type} for {symbol}")
                    model_path = self.get_model_path(symbol, model_type)

                    if os.path.exists(model_path):
                        model = self.load_model(model_type, model_path)
                        logger.info(f"Loaded existing {model_type} model for {symbol}")
                    else:
                        # Train new model với dữ liệu đã scale
                        if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                            model = self.train_nn_model(X_train, y_train_scaled.flatten(), 
                                                      X_test, y_test_scaled.flatten(), model_type)
                        else:
                            X_train_flat = X_train.reshape(X_train.shape[0], -1)
                            model = self.train_ml_model(X_train_flat, y_train_scaled.flatten(), model_type)
                        
                        self.save_model(model, model_path)

                    trained_models[model_type] = model

                    # Make predictions
                    if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                        y_pred_scaled = model.predict(X_test)
                    else:
                        X_test_flat = X_test.reshape(X_test.shape[0], -1)
                        y_pred_scaled = model.predict(X_test_flat)

                    # Inverse transform predictions
                    y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    all_predictions[model_type] = y_pred

                    # Calculate metrics với dữ liệu gốc
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Calculate MAPE
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                    
                    metrics = {
                        'model_name': model_type,
                        'RMSE': rmse,
                        'MAE': mae,
                        'R2': r2,
                        'MAPE': mape
                    }
                    all_metrics.append(metrics)
                    logger.info(f"Metrics for {model_type}: {metrics}")

                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
                    continue

            # Create ensemble predictions
            if len(all_predictions) > 1:
                ensemble_pred = self.create_ensemble_prediction(all_predictions)
                all_predictions['Ensemble'] = ensemble_pred
                
                # Calculate ensemble metrics với dữ liệu gốc
                mse = mean_squared_error(y_test, ensemble_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, ensemble_pred)
                r2 = r2_score(y_test, ensemble_pred)
                mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
                
                all_metrics.append({
                    'model_name': 'Ensemble',
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape
                })

            # Convert metrics to DataFrame
            results_df = pd.DataFrame(all_metrics)
            
            # Sort results by RMSE (ascending)
            results_df = results_df.sort_values('RMSE')
            
            # Visualize results
            self.visualize_results(results_df, symbol)
            
            # Visualize predictions
            self.visualize_predictions(y_test, all_predictions, symbol)

            return results_df, all_predictions, trained_models

        except Exception as e:
            logger.error(f"Error in train_models: {e}")
            raise

    def preprocess_data(self, X, y, training=True):
        try:
            # Reshape X for processing if needed
            X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        
            if training:
                # Use IQR method for outlier detection
                Q1 = np.percentile(X_flat, 25, axis=0)
                Q3 = np.percentile(X_flat, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            
                # Create mask for non-outlier data
                mask = np.all((X_flat >= lower_bound) & (X_flat <= upper_bound), axis=1)
            
                # If too many samples would be removed, adjust bounds
                if np.sum(mask) < len(mask) * 0.9:  # if would remove more than 10%
                    logger.warning("Too many outliers detected, adjusting bounds...")
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    mask = np.all((X_flat >= lower_bound) & (X_flat <= upper_bound), axis=1)
            else:
                mask = np.ones(len(X_flat), dtype=bool)
        
            X_clean = X[mask]
            y_clean = y[mask]
        
            # Apply power transformation to target variable
            if training:
                self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                y_transformed = self.power_transformer.fit_transform(y_clean.reshape(-1, 1)).flatten()
            else:
                if self.power_transformer is None:
                    raise ValueError("Power transformer not fitted. Call with training=True first.")
                y_transformed = self.power_transformer.transform(y_clean.reshape(-1, 1)).flatten()
        
            logger.debug(f"Preprocessed data shapes - X: {X_clean.shape}, y: {y_transformed.shape}")
            return X_clean, y_transformed
        
        except Exception as e:
            logger.error(f"Error in preprocessing data: {e}")
            raise

    def scale_data_predict(self, X):
        """Scale data for prediction"""
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            X_reshaped = X.reshape(-1, self.num_features)
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_reshaped = X.reshape(-1, self.num_features)
            X_scaled = self.scaler.transform(X_reshaped)
    
        return X_scaled.reshape(X.shape)

    def create_ensemble_prediction(self, predictions):
        """Create weighted ensemble predictions"""
        weights = {
            'LSTM': 0.20,
            'GRU': 0.20,
            'BiLSTM': 0.20,
            'TCN': 0.15,
            'Transformer': 0.10,
            'XGBoost': 0.08,
            'RF': 0.07
        }
    
        weighted_sum = np.zeros_like(list(predictions.values())[0])
        weight_sum = 0
    
        for model_type, pred in predictions.items():
            if model_type in weights:
                weighted_sum += pred * weights[model_type]
                weight_sum += weights[model_type]
    
        return weighted_sum / weight_sum if weight_sum > 0 else weighted_sum

    def check_shapes(self, X, y, model_type):
        """Check and log shapes of input data"""
        logger.info(f"Model type: {model_type}")
        logger.info(f"X shape: {X.shape}")
        logger.info(f"y shape: {y.shape}")
    
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input (samples, timesteps, features), got shape {X.shape}")
    
        if len(y.shape) != 1:
            raise ValueError(f"Expected 1D target, got shape {y.shape}")
    
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) don't match")

    def build_advanced_lstm_model(self, input_shape, units, dropout_rate, optimizer):
        """Build advanced LSTM model with attention"""
        inputs = Input(shape=input_shape)
    
        # First LSTM layer
        x = LSTM(units, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
        # Second LSTM layer
        x = LSTM(units, return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
        # Self-attention mechanism
        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=units // 4,
            attention_axes=(1,)  # Attend only to the time dimension
        )(x, x)
        x = Add()([x, attention])
        x = LayerNormalization()(x)
    
        # Final LSTM layer
        x = LSTM(units)(x)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
        # Dense layers
        x = Dense(units // 2, activation='relu')(x)
        x = Dropout(dropout_rate/2)(x)
        outputs = Dense(1)(x)
    
        model = Model(inputs, outputs)
        model.compile(
            optimizer=optimizer,
            loss=Huber(),
            metrics=['mse', 'mae']
        )
    
        return model

    def build_gru_model(self, input_shape, units, dropout_rate, optimizer):
        """Build GRU model"""
        inputs = Input(shape=input_shape)
    
        # Đảm bảo units không lớn hơn số features
        units = min(units, input_shape[-1])
    
        x = GRU(units, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
        x = GRU(units // 2, return_sequences=False)(x)  # Removed return_sequences=True
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
        x = Dense(units//4, activation='relu')(x)
        outputs = Dense(1)(x)
    
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss=Huber())
    
        return model

    def build_bilstm_model(self, input_shape, units, dropout_rate, optimizer):
        """Build Bidirectional LSTM model"""
        inputs = Input(shape=input_shape)
    
        # Ensure units doesn't exceed input dimensions
        units = min(units, input_shape[-1])
    
        x = Bidirectional(LSTM(units, return_sequences=True))(inputs)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
        x = Bidirectional(LSTM(units//2, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
        x = GlobalAveragePooling1D()(x)  # Add this layer
        x = Dense(units//2, activation='relu')(x)
        outputs = Dense(1)(x)
    
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss=Huber())
    
        return model

    def build_tcn_model(self, input_shape, nb_filters, kernel_size, dropout_rate, optimizer):
        """Build TCN model"""
        inputs = Input(shape=input_shape)
    
        # Ensure nb_filters doesn't exceed input dimensions
        nb_filters = min(nb_filters, input_shape[-1])
    
        x = TCN(
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=2,
            dilations=[1, 2, 4, 8, 16],
            padding='causal',
            use_skip_connections=True,
            dropout_rate=dropout_rate,
            return_sequences=True  # Thêm dòng này
        )(inputs)
    
        x = GlobalAveragePooling1D()(x)
        x = Dense(nb_filters//2, activation='relu')(x)
        outputs = Dense(1)(x)
    
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss=Huber())
    
        return model

    def build_transformer_model(self, input_shape, num_heads, key_dim, dropout_rate, optimizer):
        """Build Transformer model with reduced complexity"""
        inputs = Input(shape=input_shape)
    
        # Giảm độ phức tạp bằng cách thêm projection layer
        x = Dense(min(key_dim, 32), activation='linear')(inputs)
    
        # Multi-head attention với số heads và key_dim nhỏ hơn
        attention_output = MultiHeadAttention(
            num_heads=min(num_heads, 4),  # Giới hạn số heads
            key_dim=min(key_dim, 32),     # Giới hạn key dimension
            dropout=dropout_rate
        )(x, x)
    
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
        # Feed-forward network với số units nhỏ hơn
        x = Dense(min(key_dim * 2, 64), activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(min(key_dim, 32))(x)
    
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)
    
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss=Huber())
    
        return model

    def train_nn_model(self, X_train, y_train, X_val, y_val, model_type):
        """Train neural network models with hyperparameter optimization"""
        input_shape = (self.time_step, X_train.shape[-1])
        logger.info(f"Input shape for {model_type}: {input_shape}")

        def create_model(trial):
            """Create model with trial parameters"""
            units = trial.suggest_int('units', 32, min(256, input_shape[-1]))
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            optimizer = Adam(learning_rate=learning_rate)

            if model_type == 'LSTM':
                return self.build_advanced_lstm_model(input_shape, units, dropout_rate, optimizer)
            elif model_type == 'GRU':
                return self.build_gru_model(input_shape, units, dropout_rate, optimizer)
            elif model_type == 'BiLSTM':
                return self.build_bilstm_model(input_shape, units, dropout_rate, optimizer)
            elif model_type == 'TCN':
                nb_filters = trial.suggest_int('nb_filters', 32, min(256, input_shape[-1]))
                kernel_size = trial.suggest_int('kernel_size', 2, 8)
                return self.build_tcn_model(input_shape, nb_filters, kernel_size, dropout_rate, optimizer)
            elif model_type == 'Transformer':
                num_heads = trial.suggest_categorical('num_heads', [4, 8])
                key_dim = trial.suggest_int('key_dim', 32, min(128, input_shape[-1]))
                return self.build_transformer_model(input_shape, num_heads, key_dim, dropout_rate, optimizer)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        def objective(trial):
            try:
                model = create_model(trial)
            
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7),
                    TFKerasPruningCallback(trial, 'val_loss')
                ]

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=self.batch_size,
                    callbacks=callbacks,
                    verbose=0
                )

                return min(history.history['val_loss'])

            except Exception as e:
                logger.error(f"Trial failed: {e}")
                raise optuna.exceptions.TrialPruned()

        # Create and run optimization study
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(objective, n_trials=self.n_trials, catch=(ValueError,))
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

        if len(study.trials) == 0:
            raise ValueError("No successful trials completed")

        # Get best parameters and train final model
        best_trial = study.best_trial
        logger.info(f"Best parameters for {model_type}: {best_trial.params}")

        # Build final model with best parameters
        final_model = create_model(best_trial)

        # Final training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)
        ]

        final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return final_model

    def train_ml_model(self, X_train, y_train, model_type):
        """Train machine learning models with reduced memory usage"""
        try:
            # Tạo validation split
            tscv = TimeSeriesSplit(n_splits=5)
            for train_idx, val_idx in tscv.split(X_train):
                X_train_split, X_val = X_train[train_idx], X_train[val_idx]
                y_train_split, y_val = y_train[train_idx], y_train[val_idx]
                break  # Chỉ lấy split đầu tiên
        
            def objective(trial):
                try:
                    if model_type == 'XGBoost':
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                            'max_depth': trial.suggest_int('max_depth', 3, 8),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                            'gamma': trial.suggest_float('gamma', 0, 5),
                            'tree_method': 'hist',  # Sử dụng histogram-based algorithm
                            'max_bin': 256  # Giảm số bins trong histogram
                        }
                        model = XGBRegressor(**params, random_state=42)
                    elif model_type == 'RF':
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                            'max_depth': trial.suggest_int('max_depth', 5, 15),
                            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                        }
                        model = RandomForestRegressor(**params, random_state=42)

                    # Fit model với validation data
                    model.fit(X_train_split, y_train_split)
                    y_pred = model.predict(X_val)
                
                    # Tính RMSE thủ công
                    mse = mean_squared_error(y_val, y_pred)
                    rmse = np.sqrt(mse)
                    return rmse

                except Exception as e:
                    logger.error(f"Trial failed: {e}")
                    raise optuna.exceptions.TrialPruned()

            # Tạo và chạy optimization study
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=self.n_trials, catch=(ValueError,))

            # Train final model với best parameters
            if model_type == 'XGBoost':
                final_model = XGBRegressor(
                    **study.best_params,
                    tree_method='hist',
                    max_bin=256,
                    random_state=42
                )
            else:
                final_model = RandomForestRegressor(
                    **study.best_params,
                    random_state=42
                )

            # Train trên toàn bộ training data
            final_model.fit(X_train, y_train)
            return final_model

        except Exception as e:
            logger.error(f"Error in train_ml_model: {e}")
            raise

    def predict(self, model, model_type, X):
        """Make predictions using the trained model"""
        try:
            if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                y_pred = model.predict(X)
            else:
                if len(X.shape) > 2:
                    X = X.reshape(X.shape[0], -1)
                y_pred = model.predict(X)
                y_pred = y_pred.reshape(-1, 1)
            
            return y_pred.flatten()
            
        except Exception as e:
            logger.error(f"Prediction error for {model_type}: {e}")
            raise

    def visualize_results(self, results_df, symbol):
        """Visualize model comparison results"""
        try:
            plt.figure(figsize=(12, 6))
            metrics = ['RMSE', 'MAE', 'R2']
        
            for metric in metrics:
                if metric in results_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=results_df, x='model_name', y=metric)
                    plt.title(f'{metric} Comparison for {symbol}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.model_dir, f'{symbol}_{metric}_comparison.png'))
                    plt.close()
                
        except Exception as e:
            logger.error(f"Error in visualize_results: {e}")

    def get_model_path(self, symbol, model_type):
        """Get the path for saving/loading models"""
        if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
            extension = 'keras'
        else:
            extension = 'pkl'
        return os.path.join(self.model_dir, f'{symbol}_{model_type}_model.{extension}')

    def save_model(self, model, path):
        """Save the trained model"""
        try:
            if isinstance(model, tf.keras.Model):
                model.save(path)
            else:
                joblib.dump(model, path)
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}")
            raise

    def load_model(self, model_type, path):
        """Load a trained model"""
        try:
            if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                return load_model(path)
            else:
                return joblib.load(path)
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            raise

    def visualize_predictions(self, y_true, predictions, symbol):
        """Visualize predictions from different models"""
        try:
            # Create figure for predictions comparison
            plt.figure(figsize=(15, 8))
            
            # Plot actual values
            plt.plot(y_true, label='Actual', color='black', alpha=0.7, linewidth=2)
            
            # Plot predictions from each model with different colors
            colors = plt.cm.Set2(np.linspace(0, 1, len(predictions)))
            for (model_name, y_pred), color in zip(predictions.items(), colors):
                plt.plot(y_pred, label=f'{model_name}', alpha=0.6, linewidth=1.5, color=color)
            
            plt.title(f'Price Predictions Comparison for {symbol}')
            plt.xlabel('Time Steps')
            plt.ylabel('Price')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save predictions comparison plot
            pred_path = os.path.join(self.model_dir, f'{symbol}_predictions_comparison.png')
            plt.savefig(pred_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Create figure for error analysis
            plt.figure(figsize=(15, 8))
            
            # Calculate and plot absolute errors
            for (model_name, y_pred), color in zip(predictions.items(), colors):
                error = np.abs(y_true - y_pred)
                plt.plot(error, label=f'{model_name}', alpha=0.6, linewidth=1.5, color=color)
            
            plt.title(f'Prediction Error Analysis for {symbol}')
            plt.xlabel('Time Steps')
            plt.ylabel('Absolute Error')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save error analysis plot
            error_path = os.path.join(self.model_dir, f'{symbol}_error_analysis.png')
            plt.savefig(error_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Create performance comparison bar plot
            plt.figure(figsize=(12, 6))
            metrics_df = pd.DataFrame([{
                'Model': model_name,
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            } for model_name, y_pred in predictions.items()])
            
            metrics_df = metrics_df.sort_values('RMSE')
            
            # Plot RMSE comparison
            plt.subplot(131)
            sns.barplot(data=metrics_df, x='Model', y='RMSE')
            plt.xticks(rotation=45)
            plt.title('RMSE Comparison')
            
            # Plot MAE comparison
            plt.subplot(132)
            sns.barplot(data=metrics_df, x='Model', y='MAE')
            plt.xticks(rotation=45)
            plt.title('MAE Comparison')
            
            # Plot MAPE comparison
            plt.subplot(133)
            sns.barplot(data=metrics_df, x='Model', y='MAPE')
            plt.xticks(rotation=45)
            plt.title('MAPE Comparison')
            
            plt.tight_layout()
            
            # Save performance comparison plot
            perf_path = os.path.join(self.model_dir, f'{symbol}_performance_comparison.png')
            plt.savefig(perf_path, bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            logger.error(f"Error in visualize_predictions: {e}")
