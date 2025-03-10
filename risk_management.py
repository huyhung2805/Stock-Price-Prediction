import numpy as np
import pandas as pd
from scipy import stats

class RiskManager:
    def __init__(self):
        self.confidence_level = 0.95
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

    def calculate_value_at_risk(self, returns, confidence_level=None):
        """Calculate Value at Risk"""
        if confidence_level is None:
            confidence_level = self.confidence_level
        return np.percentile(returns, (1 - confidence_level) * 100)

    def calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe Ratio"""
        excess_returns = returns - self.risk_free_rate/252  # Daily adjustment
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    def calculate_max_drawdown(self, prices):
        """Calculate Maximum Drawdown"""
        rolling_max = prices.expanding().max()
        drawdowns = prices/rolling_max - 1
        return drawdowns.min()

    def calculate_risk_metrics(self, predictions, historical_data):
        """Calculate comprehensive risk metrics"""
        returns = historical_data['Close'].pct_change().dropna()
        
        return {
            'value_at_risk': self.calculate_value_at_risk(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(historical_data['Close']),
            'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }