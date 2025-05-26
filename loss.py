import torch
import torch.nn as nn
import torch.nn.functional as F

# # Daily
# class MeanVarianceLoss(nn.Module):
#     def __init__(self, risk_aversion=1.0):
#         super(MeanVarianceLoss, self).__init__()
#         self.risk_aversion = risk_aversion
        
#     def forward(self, returns, weights, batch_x):
#         """
#         Module 2: Mean-Variance Portfolio Loss
#         Input: 
#             returns: [B, N] - next period returns
#             weights: [B, N] - portfolio weights
#             batch_x: [B, N, T] - historical return series for covariance calculation
#         Output: 
#             loss: scalar - negative of mean-variance objective
#         """
#         # Calculate portfolio mean return using next period returns
#         portfolio_return = torch.sum(returns * weights, dim=1)  # [B]
        
#         # Calculate sample covariance matrix using historical data (batch_x)
#         B, N, T = batch_x.shape
#         mean = batch_x.mean(dim=2, keepdim=True)  # [B, N, 1]
#         X_centered = batch_x - mean  # [B, N, T]
#         cov_matrix = torch.matmul(X_centered, X_centered.transpose(1, 2)) / (T - 1)  # [B, N, N]
        
#         # Calculate portfolio variance using the sample covariance matrix
#         portfolio_var = torch.matmul(weights.unsqueeze(1), torch.matmul(cov_matrix, weights.unsqueeze(2))).squeeze()
        
#         # Mean-Variance objective: maximize (return - risk_aversion * variance)
#         loss = -(portfolio_return - self.risk_aversion * portfolio_var)
        
#         return loss.mean()

# class MaxSharpeRatioLoss(nn.Module):
#     def __init__(self, risk_free_rate=0.0):
#         super(MaxSharpeRatioLoss, self).__init__()
#         self.risk_free_rate = risk_free_rate
        
#     def forward(self, returns, weights, batch_x):
#         """
#         Module 3: Maximum Sharpe Ratio Portfolio Loss
#         Input:
#             returns: [B, N] - next period returns
#             weights: [B, N] - portfolio weights
#             batch_x: [B, N, T] - historical return series for covariance calculation
#         Output:
#             loss: scalar - negative of Sharpe ratio
#         """
#         # Calculate portfolio mean return using next period returns
#         portfolio_return = torch.sum(returns * weights, dim=1)  # [B]
        
#         # Calculate sample covariance matrix using historical data (batch_x)
#         B, N, T = batch_x.shape
#         mean = batch_x.mean(dim=2, keepdim=True)  # [B, N, 1]
#         X_centered = batch_x - mean  # [B, N, T]
#         cov_matrix = torch.matmul(X_centered, X_centered.transpose(1, 2)) / (T - 1)  # [B, N, N]
        
#         # Calculate portfolio variance using the sample covariance matrix
#         portfolio_var = torch.matmul(weights.unsqueeze(1), torch.matmul(cov_matrix, weights.unsqueeze(2))).squeeze()
#         portfolio_std = torch.sqrt(portfolio_var + 1e-6)
        
#         # Calculate excess return
#         excess_return = portfolio_return - self.risk_free_rate
        
#         # Sharpe ratio = excess_return / std_dev
#         sharpe_ratio = excess_return / (portfolio_std + 1e-6)
        
#         # Loss is negative Sharpe ratio (we want to maximize Sharpe ratio)
#         loss = -sharpe_ratio
        
#         return loss.mean()


# Annualized

class MeanVarianceLoss(nn.Module):
    def __init__(self, risk_aversion=1.0, periods_per_year=252):
        super(MeanVarianceLoss, self).__init__()
        self.risk_aversion = risk_aversion
        self.periods_per_year = periods_per_year
        
    def forward(self, returns, weights, batch_x):
        """
        Module 2: Mean-Variance Portfolio Loss (Annualized)
        Input: 
            returns: [B, N] - next period returns
            weights: [B, N] - portfolio weights
            batch_x: [B, N, T] - historical return series for covariance calculation
        Output: 
            loss: scalar - negative of mean-variance objective
        """
        # Calculate portfolio mean return using next period returns
        portfolio_return = torch.sum(returns * weights, dim=1)  # [B]
        # Annualize returns
        portfolio_return = portfolio_return * self.periods_per_year
        
        # Calculate sample covariance matrix using historical data (batch_x)
        B, N, T = batch_x.shape
        mean = batch_x.mean(dim=2, keepdim=True)  # [B, N, 1]
        X_centered = batch_x - mean  # [B, N, T]
        cov_matrix = torch.matmul(X_centered, X_centered.transpose(1, 2)) / (T - 1)  # [B, N, N]
        # Annualize covariance matrix
        cov_matrix = cov_matrix * self.periods_per_year
        
        # Calculate portfolio variance using the sample covariance matrix
        portfolio_var = torch.matmul(weights.unsqueeze(1), torch.matmul(cov_matrix, weights.unsqueeze(2))).squeeze()
        
        # Mean-Variance objective: maximize (return - risk_aversion * variance)
        loss = -(portfolio_return - self.risk_aversion * portfolio_var)
        
        return loss.mean()

class MaxSharpeRatioLoss(nn.Module):
    def __init__(self, risk_free_rate=0.0, periods_per_year=252):
        super(MaxSharpeRatioLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
    def forward(self, returns, weights, batch_x):
        """
        Module 3: Maximum Sharpe Ratio Portfolio Loss (Annualized)
        Input:
            returns: [B, N] - next period returns
            weights: [B, N] - portfolio weights
            batch_x: [B, N, T] - historical return series for covariance calculation
        Output:
            loss: scalar - negative of Sharpe ratio
        """
        # Calculate portfolio mean return using next period returns
        portfolio_return = torch.sum(returns * weights, dim=1)  # [B]
        # Annualize returns
        portfolio_return = portfolio_return * self.periods_per_year
        
        # Calculate sample covariance matrix using historical data (batch_x)
        B, N, T = batch_x.shape
        mean = batch_x.mean(dim=2, keepdim=True)  # [B, N, 1]
        X_centered = batch_x - mean  # [B, N, T]
        cov_matrix = torch.matmul(X_centered, X_centered.transpose(1, 2)) / (T - 1)  # [B, N, N]
        # Annualize covariance matrix
        cov_matrix = cov_matrix * self.periods_per_year
        
        # Calculate portfolio variance using the sample covariance matrix
        portfolio_var = torch.matmul(weights.unsqueeze(1), torch.matmul(cov_matrix, weights.unsqueeze(2))).squeeze()
        portfolio_std = torch.sqrt(portfolio_var + 1e-6)
        
        # Annualize risk-free rate
        annualized_rf = self.risk_free_rate * self.periods_per_year
        
        # Calculate excess return
        excess_return = portfolio_return - annualized_rf
        
        # Sharpe ratio = excess_return / std_dev
        sharpe_ratio = excess_return / (portfolio_std + 1e-6)
        
        # Loss is negative Sharpe ratio (we want to maximize Sharpe ratio)
        loss = -sharpe_ratio
        
        return loss.mean()