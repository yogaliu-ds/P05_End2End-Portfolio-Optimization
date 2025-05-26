'''
An Example Module: Try the simple version of the model without trainging the model.
'''

import torch
import numpy as np
import pandas as pd
from score_block import ScoreBlock
from portfolio_block import PortfolioBlock
from loss import MeanVarianceLoss, MaxSharpeRatioLoss
from evaluation_metrics import PortfolioMetrics

def main():
    # Example usage
    batch_size = 32
    num_assets = 331  # SP500 data dimension
    time_steps = 252
    
    # Generate random return series
    returns = torch.randn(batch_size, num_assets, time_steps)
    
    # Initialize modules
    score_block = ScoreBlock(input_dim=num_assets, hidden_dim=64)  # Fixed input_dim to match num_assets
    portfolio_block = PortfolioBlock()
    mv_loss = MeanVarianceLoss(risk_aversion=1.0)
    sharpe_loss = MaxSharpeRatioLoss(risk_free_rate=0.0)
    
    # Get scores from ScoreBlock
    scores = score_block(returns)
    
    # Get portfolio weights
    weights = portfolio_block(scores)
    
    # Calculate losses
    mv_loss_value = mv_loss(returns, weights)
    sharpe_loss_value = sharpe_loss(returns, weights)
    
    # Calculate metrics
    metrics = PortfolioMetrics()
        
    portfolio_metrics = {
        'Expected Return': metrics.expected_return(returns, weights),
        'Standard Deviation': metrics.standard_deviation(returns, weights),
        'Sharpe Ratio': metrics.sharpe_ratio(returns, weights),
        'Drawdown': metrics.drawdown(returns, weights),
        'Sortino Ratio': metrics.sortino_ratio(returns, weights),
        'Maximum Drawdown': metrics.maximum_drawdown(returns, weights),
        'Positive Return %': metrics.positive_return_percentage(returns, weights)
    }
    
    # Print results
    print("Portfolio Metrics:")
    for metric_name, value in portfolio_metrics.items():
        if isinstance(value, torch.Tensor):
            print(f"{metric_name}: {value.mean().item():.4f}")
        else:
            print(f"{metric_name}: {value:.4f}")
            
    print("\nLoss Values:")
    print(f"Mean-Variance Loss: {mv_loss_value.mean().item():.4f}")
    print(f"Sharpe Ratio Loss: {sharpe_loss_value.mean().item():.4f}")

if __name__ == "__main__":
    main()