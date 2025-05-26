import numpy as np
import torch

class PortfolioMetrics:
    @staticmethod
    def expected_return(returns, weights):
        """Calculate expected return over a period"""
        # returns: [B, N], weights: [B, N]
        portfolio_return = torch.sum(returns * weights, dim=1)  # [B]
        # Annualize the expected return (assuming 252 trading days)
        return portfolio_return * 252  # [B]
    
    @staticmethod
    def standard_deviation(returns, weights, batch_x):
        """Calculate portfolio standard deviation over a period"""
        # batch_x: [B, N, T], returns: [B, N]
        B, N, T1 = batch_x.shape
        _, _, T2 = returns.unsqueeze(-1).shape
        
        # Concatenate historical data with current returns
        combined_returns = torch.cat([batch_x, returns.unsqueeze(-1)], dim=2)  # [B, N, T1+T2]
        
        # Calculate covariance matrix using combined data
        mean = combined_returns.mean(dim=2, keepdim=True)  # [B, N, 1]
        X_centered = combined_returns - mean  # [B, N, T1+T2]
        cov_matrix = torch.matmul(X_centered, X_centered.transpose(1, 2)) / (T1 + T2 - 1)  # [B, N, N]
        
        # Calculate portfolio variance
        portfolio_var = torch.matmul(weights.unsqueeze(1), torch.matmul(cov_matrix, weights.unsqueeze(2))).squeeze()
        # Annualize the standard deviation (assuming 252 trading days)
        annualized_std = torch.sqrt(portfolio_var * 252 + 1e-6)  # [B]
        return annualized_std
    
    @staticmethod
    def sharpe_ratio(returns, weights, batch_x, risk_free_rate=0.0):
        """Calculate Sharpe ratio over a period"""
        exp_return = PortfolioMetrics.expected_return(returns, weights)
        std_dev = PortfolioMetrics.standard_deviation(returns, weights, batch_x)
        return (exp_return - risk_free_rate) / (std_dev + 1e-6)
    
   
    @staticmethod
    def sortino_ratio(returns, weights, batch_x, risk_free_rate=0.0, target_return=0.0):
        """Calculate Sortino ratio over a period"""
        # returns: [B, N], weights: [B, N]
        portfolio_return = torch.sum(returns * weights, dim=1)  # [B]
        exp_return = PortfolioMetrics.expected_return(returns, weights)  # Already annualized
        
        # Calculate downside deviation using historical data
        B, N, T = batch_x.shape
        portfolio_returns_hist = torch.sum(batch_x * weights.unsqueeze(-1), dim=1)  # [B, T]
        downside_returns = torch.minimum(portfolio_returns_hist - target_return, torch.zeros_like(portfolio_returns_hist))
        # Annualize the downside std (assuming 252 trading days)
        downside_std = torch.sqrt(torch.mean(downside_returns ** 2, dim=1) * 252)
        
        return (exp_return - risk_free_rate) / (downside_std + 1e-6)

    @staticmethod
    def maximum_drawdown(returns, weights):
        """
        Compute maximum drawdown from time series returns and weights.
        
        Inputs:
            returns: [T, N] - asset returns at each time step (log returns)
            weights: [T, N] - portfolio weights at each time step
            
        Returns:
            max_drawdown: scalar - maximum drawdown
        """
        # [T] - portfolio returns per time step
        portfolio_returns = torch.sum(returns * weights, dim=1)

        # Convert log returns to simple returns
        simple_returns = torch.exp(portfolio_returns)

        # [T] - cumulative return over time
        cumulative_return = torch.cumprod(simple_returns, dim=0)

        # [T] - running max of cumulative return
        running_max = torch.cummax(cumulative_return, dim=0).values

        # [T] - drawdown at each time
        drawdown = (running_max - cumulative_return) / (running_max + 1e-6)

        # scalar - maximum drawdown
        max_drawdown = torch.max(drawdown)

        return max_drawdown

    
    @staticmethod
    def positive_return_percentage(returns, weights):
        """Calculate percentage of positive returns over a period"""
        # returns: [B, N], weights: [B, N]
        portfolio_return = torch.sum(returns * weights, dim=1)  # [B]
        percentage = (portfolio_return > 0).float().mean()
        return percentage  # [1]
    
    @staticmethod
    def frobenius_norm(matrix):
        """Calculate Frobenius norm of a matrix"""
        return torch.sqrt(torch.sum(matrix ** 2))
    
    @staticmethod
    def turnover(weights):
        """
        Calculate portfolio turnover over time.
        
        Inputs:
            weights: [T, N] - portfolio weights at each time step
            
        Returns:
            turnover: scalar - average turnover per period
        """
        # Calculate absolute weight changes between consecutive periods
        weight_changes = torch.abs(weights[1:, :] - weights[:-1, :])
        
        # Sum the absolute changes across all assets for each period
        period_turnover = torch.sum(weight_changes, dim=1)
        print(period_turnover)
        
        # Calculate average turnover per period
        avg_turnover = torch.mean(period_turnover)
        
        return avg_turnover