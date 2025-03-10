import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
import logging
import numpy as np
import pandas as pd
import argparse
from threading import Thread
import yaml
from datetime import datetime
import optuna
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.tensorflow
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from risk_management import RiskManager
from portfolio_optimization import PortfolioOptimizer

from data_processing import DataProcessor
from modeling import ModelTrainer
from prediction import Predictor
from utils import evaluate_model_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockPredictionPipeline:
    def __init__(self, config_path='config.yaml'):
        self.load_config(config_path)
        self.setup_mlflow()
        self.risk_manager = RiskManager()
        self.portfolio_optimizer = PortfolioOptimizer()

    def analyze_portfolio(self, symbols, predictions, historical_data):
        """Analyze portfolio using predictions and historical data"""
        # Calculate risk metrics
        risk_metrics = {
            symbol: self.risk_manager.calculate_risk_metrics(
                predictions[symbol], 
                historical_data[symbol]
            ) for symbol in symbols
        }

        # Calculate expected returns
        expected_returns = self.portfolio_optimizer.calculate_expected_returns(predictions)
        
        # Calculate historical returns for covariance
        historical_returns = pd.DataFrame({
            symbol: historical_data[symbol]['Close'].pct_change()
            for symbol in symbols
        }).dropna()
        
        # Calculate covariance matrix
        cov_matrix = self.portfolio_optimizer.calculate_covariance_matrix(historical_returns)
        
        # Optimize portfolio
        optimal_weights = self.portfolio_optimizer.optimize_portfolio(
            expected_returns, 
            cov_matrix
        )
        
        # Generate efficient frontier
        efficient_frontier = self.portfolio_optimizer.generate_efficient_frontier(
            expected_returns, 
            cov_matrix
        )

        return {
            'risk_metrics': risk_metrics,
            'optimal_weights': dict(zip(symbols, optimal_weights)),
            'efficient_frontier': efficient_frontier,
            'portfolio_return': self.portfolio_optimizer.portfolio_return(
                optimal_weights, 
                expected_returns
            ),
            'portfolio_volatility': self.portfolio_optimizer.portfolio_volatility(
                optimal_weights, 
                cov_matrix
            )
        }
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Set paths
        self.data_dir = self.config['paths']['data_dir']
        self.model_dir = self.config['paths']['model_dir']
        self.results_dir = self.config['paths']['results_dir']
        
        # Create directories if they don't exist
        for directory in [self.model_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
    def setup_mlflow(self):
        """Configure MLflow for experiment tracking"""
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def train_models(self, symbol, args):
        """Train and evaluate models for a single symbol"""
        try:
            logger.info(f"Starting training pipeline for {symbol}")
            
            # Initialize data processor
            data_processor = DataProcessor(self.data_dir, symbol)
            df = data_processor.load_data()
            if df is None or df.empty:
                logger.error(f"No data available for {symbol}")
                return None
                
            # Add technical indicators and features
            from feature_engineering import add_technical_indicators
            df = add_technical_indicators(df.reset_index())
            
            # Prepare data for modeling
            scaled_features, target = data_processor.preprocess_data(df)
            X, y = data_processor.create_sequences(scaled_features, 
                                                 self.config['model']['time_step'],
                                                 data_processor.feature_columns.index('Close'))
            
            # Split data
            split_index = int(len(X) * self.config['data']['train_split'])
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Initialize model trainer
            trainer = ModelTrainer(
                model_dir=self.model_dir,
                time_step=self.config['model']['time_step'],
                num_features=X.shape[2]
            )
            
            # Set training parameters
            trainer.epochs = args.epochs
            trainer.batch_size = args.batch_size
            trainer.n_trials = args.n_trials
            
            # Train and evaluate models
            results_df, predictions, trained_models = trainer.train_models(
                X_train, y_train, X_test, y_test, symbol
            )
            
            # Hiển thị kết quả chi tiết
            print("\n" + "="*50)
            print(f"Training Results for {symbol}")
            print("="*50)
            
            # Format và hiển thị DataFrame kết quả
            pd.set_option('display.float_format', lambda x: '%.4f' % x)
            print("\nModel Performance Metrics:")
            print(results_df.to_string(index=False))
            
            # Hiển thị model tốt nhất
            best_model = results_df.iloc[0]
            print("\nBest Performing Model:")
            print(f"Model: {best_model['model_name']}")
            print(f"RMSE: {best_model['RMSE']:.4f}")
            print(f"MAE: {best_model['MAE']:.4f}")
            print(f"R2: {best_model['R2']:.4f}")
            print(f"MAPE: {best_model['MAPE']:.4f}%")
            
            # Log metrics to MLflow
            with mlflow.start_run(run_name=f"{symbol}_{datetime.now().strftime('%Y%m%d')}"):
                mlflow.log_params({
                    "symbol": symbol,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "n_trials": args.n_trials
                })
                
                for _, row in results_df.iterrows():
                    mlflow.log_metrics({
                        f"{row['model_name']}_RMSE": row['RMSE'],
                        f"{row['model_name']}_MAE": row['MAE'],
                        f"{row['model_name']}_R2": row['R2'],
                        f"{row['model_name']}_MAPE": row['MAPE']
                    })
            
            # Save results
            results_path = os.path.join(self.results_dir, f"{symbol}_results.csv")
            results_df.to_csv(results_path)
            print(f"\nResults saved to: {results_path}")
            
            # Log artifacts
            mlflow.log_artifact(results_path)
            mlflow.log_artifact(os.path.join(self.model_dir, f'{symbol}_predictions_comparison.png'))
            mlflow.log_artifact(os.path.join(self.model_dir, f'{symbol}_error_analysis.png'))
            mlflow.log_artifact(os.path.join(self.model_dir, f'{symbol}_performance_comparison.png'))
            
            print("\nVisualization files generated:")
            print(f"1. {symbol}_predictions_comparison.png")
            print(f"2. {symbol}_error_analysis.png")
            print(f"3. {symbol}_performance_comparison.png")
            
            return results_df, predictions, trained_models
            
        except Exception as e:
            logger.error(f"Error in training pipeline for {symbol}: {e}")
            return None

    def train_models_parallel(self, symbols, args):
        """Train models in parallel for multiple symbols"""
        max_workers = multiprocessing.cpu_count() - 1
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.train_models, symbol, args): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error training models for {symbol}: {e}")
                    results[symbol] = None
                    
        return results

    def run_parallel_backtesting(self, symbols, model_type, start_date=None, end_date=None):
        """Run backtesting in parallel"""
        with ThreadPoolExecutor() as executor:
            future_to_symbol = {
                executor.submit(self.run_backtesting, symbol, model_type, start_date, end_date): symbol 
                for symbol in symbols
            }
            
            results = {}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error in backtesting for {symbol}: {e}")
                    results[symbol] = None
                    
        return results

    def run_backtesting(self, symbol, model_type, start_date=None, end_date=None):
        """Perform backtesting for a specific model"""
        try:
            predictor = Predictor(self.model_dir, self.data_dir, self.config['model']['time_step'])
            data_processor = DataProcessor(self.data_dir, symbol)
            df = data_processor.load_data()
            
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            # Implement backtesting logic
            window_size = self.config['backtest']['window_size']
            results = []
            
            for i in range(window_size, len(df)):
                train_data = df.iloc[i-window_size:i]
                test_data = df.iloc[i:i+1]
                
                # Make prediction
                pred = predictor.predict(symbol, model_type)
                
                # Calculate metrics
                actual = test_data['Close'].values[0]
                predicted = pred[0]
                
                results.append({
                    'Date': test_data.index[0],
                    'Actual': actual,
                    'Predicted': predicted,
                    'Error': actual - predicted
                })
                
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(self.results_dir, f"{symbol}_{model_type}_backtest.csv"))
            return results_df
            
        except Exception as e:
            logger.error(f"Error in backtesting for {symbol}: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_type', type=str, default='all', help='Model type to train')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = StockPredictionPipeline(args.config)
    
    # Get list of symbols
    data_files = [f for f in os.listdir(pipeline.data_dir) if f.endswith('_historical_data.csv')]
    symbols = [f.replace('_historical_data.csv', '') for f in data_files]
    
    overall_results = {}
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}")
        
        # Train models
        results = pipeline.train_models(symbol, args)
        if results is not None:
            overall_results[symbol] = results
            
        # Run backtesting if requested
        if args.backtest and results is not None:
            best_model = results[0].loc[results[0]['RMSE'].idxmin(), 'model_name']
            backtest_results = pipeline.run_backtesting(symbol, best_model)
            if backtest_results is not None:
                logger.info(f"Backtesting results saved for {symbol}")

    # Aggregate results
    if overall_results:
        all_results = pd.concat(overall_results.values(), keys=overall_results.keys())
        all_results.to_csv(os.path.join(pipeline.results_dir, 'overall_results.csv'))
        logger.info("Training completed. Results saved.")
    else:
        logger.warning("No results to save.")

if __name__ == '__main__':
    main()