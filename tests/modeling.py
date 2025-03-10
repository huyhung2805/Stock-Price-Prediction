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
            # Preprocess data
            X_train, y_train = self.preprocess_data(X_train, y_train)
            X_test, y_test = self.preprocess_data(X_test, y_test, training=False)
        
            # Scale data
            X_train_scaled = self.scale_data_predict(X_train)
            X_test_scaled = self.scale_data_predict(X_test)
        
            # Check shapes
            self.check_shapes(X_train_scaled, y_train, "Training data")
            self.check_shapes(X_test_scaled, y_test, "Test data")
        
            logger.debug(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
        
            model_types = ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer', 'XGBoost', 'RF']
            results = {}
            predictions = {}
            all_results = []

            # Scale data for deep learning models
            X_train_scaled = self.scale_data_predict(X_train)
            X_test_scaled = self.scale_data_predict(X_test)

            for model_type in model_types:
                try:
                    logger.info(f"Training {model_type} for {symbol}")
                    model_path = self.get_model_path(symbol, model_type)

                    if os.path.exists(model_path):
                        model = self.load_model(model_type, model_path)
                        logger.info(f"Loaded existing {model_type} model for {symbol}")
                    else:
                        # Train new model
                        if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                            model = self.train_nn_model(X_train_scaled, y_train, 
                                                      X_test_scaled, y_test, model_type)
                        else:
                            # Flatten data for traditional ML models
                            X_train_flat = X_train.reshape(X_train.shape[0], -1)
                            model = self.train_ml_model(X_train_flat, y_train, model_type)
                    
                        self.save_model(model, model_path)

                    # Make predictions
                    if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                        y_pred = model.predict(X_test_scaled)
                    else:
                        X_test_flat = X_test.reshape(X_test.shape[0], -1)
                        y_pred = model.predict(X_test_flat)

                    y_pred = y_pred.reshape(-1)
                    predictions[model_type] = y_pred

                    # Evaluate model
                    metrics = evaluate_model_performance(y_test, y_pred, model_type)
                    results[model_type] = metrics
                    all_results.append(metrics)

                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
                    continue

            # Create ensemble predictions if we have multiple predictions
            if len(predictions) > 1:
                ensemble_pred = self.create_ensemble_prediction(predictions)
                ensemble_metrics = evaluate_model_performance(y_test, ensemble_pred, 'Ensemble')
                results['Ensemble'] = ensemble_metrics
                all_results.append(ensemble_metrics)

            # Convert results to DataFrame
            results_df = pd.DataFrame(all_results)
        
            # Save results
            results_path = os.path.join(self.model_dir, f'{symbol}_results.csv')
            results_df.to_csv(results_path, index=False)
        
            # Create visualization
            self.visualize_results(results_df, symbol)

            return results_df, predictions

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
        """Build Transformer model"""
        inputs = Input(shape=input_shape)
    
        x = LayerNormalization()(inputs)
    
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim
        )(x, x)
    
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
        # Feed-forward network
        x = Dense(key_dim*4, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(key_dim)(x)
    
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
        """Train machine learning models with hyperparameter optimization"""
        def objective(trial):
            try:
                if model_type == 'XGBoost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                    }
                    model = XGBRegressor(**params, random_state=42)
                elif model_type == 'RF':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                    }
                    model = RandomForestRegressor(**params, random_state=42)

                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                scores = []
            
                for train_idx, val_idx in tscv.split(X_train):
                    X_t, X_v = X_train[train_idx], X_train[val_idx]
                    y_t, y_v = y_train[train_idx], y_train[val_idx]
                
                    model.fit(X_t, y_t)
                    y_pred = model.predict(X_v)
                    mse = mean_squared_error(y_v, y_pred)
                    scores.append(np.sqrt(mse))

                return np.mean(scores)

            except Exception as e:
                logger.error(f"Trial failed: {e}")
                raise optuna.exceptions.TrialPruned()

        # Create and run optimization study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)

        # Train final model with best parameters
        best_params = study.best_params
        logger.info(f"Best parameters for {model_type}: {best_params}")

        if model_type == 'XGBoost':
            final_model = XGBRegressor(**best_params, random_state=42)
        elif model_type == 'RF':
            final_model = RandomForestRegressor(**best_params, random_state=42)

        final_model.fit(X_train, y_train)
        return final_model

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
