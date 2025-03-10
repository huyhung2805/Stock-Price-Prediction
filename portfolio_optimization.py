import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate

    def calculate_expected_returns(self, predictions_dict):
        """Calculate expected returns from predictions"""
        returns = {}
        for symbol, pred in predictions_dict.items():
            returns[symbol] = (pred[-1] - pred[0]) / pred[0]
        return pd.Series(returns)

    def calculate_covariance_matrix(self, historical_returns):
        """Calculate covariance matrix from historical returns"""
        return historical_returns.cov() * 252  # Annualize covariance

    def portfolio_volatility(self, weights, cov_matrix):
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def portfolio_return(self, weights, returns):
        """Calculate portfolio return"""
        return np.sum(returns * weights)

    def optimize_portfolio(self, returns, cov_matrix, target_return=None):
        """Optimize portfolio weights using Mean-Variance Optimization"""
        num_assets = len(returns)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.portfolio_return(x, returns) - target_return
            })
        
        # Define bounds (0 to 1 for each weight)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            fun=self.portfolio_volatility,
            x0=initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_weights

    def generate_efficient_frontier(self, returns, cov_matrix, points=100):
        """Generate efficient frontier points"""
        min_return = min(returns)
        max_return = max(returns)
        target_returns = np.linspace(min_return, max_return, points)
        
        efficient_portfolios = []
        for target in target_returns:
            weights = self.optimize_portfolio(returns, cov_matrix, target)
            vol = self.portfolio_volatility(weights, cov_matrix)
            ret = self.portfolio_return(weights, returns)
            efficient_portfolios.append({
                'return': ret,
                'volatility': vol,
                'weights': weights
            })
            
        return pd.DataFrame(efficient_portfolios)