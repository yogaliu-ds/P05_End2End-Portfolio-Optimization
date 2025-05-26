import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
from torch import optim
from tqdm.notebook import tqdm
from score_block import ScoreBlock
from portfolio_block import PortfolioBlock
from loss import MeanVarianceLoss, MaxSharpeRatioLoss
from evaluation_metrics import PortfolioMetrics

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and preprocessing
def load_data():
    sdf = pd.read_csv('https://raw.githubusercontent.com/matsuda318/data/main/SP500_Price.csv', index_col=0)
    sdf2 = pd.read_csv('https://raw.githubusercontent.com/matsuda318/data/main/SP500_High.csv', index_col=0)
    sdf3 = pd.read_csv('https://raw.githubusercontent.com/matsuda318/data/main/SP500_Low.csv', index_col=0)
    
    ndt = np.array(sdf)
    ndt2 = np.array(sdf2)
    ndt3 = np.array(sdf3)
    
    dt = np.log(ndt[1:,:])-np.log(ndt[:-1,:])
    dt3 = np.log(ndt3[1:,:])-np.log(ndt3[:-1,:])
    dt2 = np.log(ndt2[1:,:])-np.log(ndt3[1:,:])
    
    return dt, dt2, dt3

def prepare_data(dt, M1=21, T0=3378, T1=4027, T2=5287, batch_size=32):
    # Prepare training data
    x_train, y_train = [], []
    for t in range(M1, T0):
        x_train.append(dt[t-M1:t,:].T)
        y_train.append(dt[t,:].T)  # Get next 252 days' returns
    x_train = torch.from_numpy(np.array(x_train)).float()
    y_train = torch.from_numpy(np.array(y_train)).float()
    
    # Prepare validation data
    x_val, y_val = [], []
    for t in range(T0, T1):
        x_val.append(dt[t-M1:t,:].T)
        y_val.append(dt[t,:].T)  # Get next 252 days' returns
    x_val = torch.from_numpy(np.array(x_val)).float()
    y_val = torch.from_numpy(np.array(y_val)).float()
    
    # Prepare test data
    x_test, y_test = [], []
    for t in range(T1, T2-252):  # Adjust range to ensure we have 252 future points
        x_test.append(dt[t-M1:t,:].T)
        y_test.append(dt[t,:].T)  # Get next 252 days' returns
    x_test = torch.from_numpy(np.array(x_test)).float()
    y_test = torch.from_numpy(np.array(y_test)).float()
    
    # Create dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    val_dataset = TensorDataset(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_dataloader, val_dataloader, test_dataloader

class PortfolioModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(PortfolioModel, self).__init__()
        self.score_block = ScoreBlock(input_dim=input_dim, hidden_dim=hidden_dim)
        self.portfolio_block = PortfolioBlock()
        
    def forward(self, x):
        scores = self.score_block(x)
        weights = self.portfolio_block(scores)
        return weights

def train_epoch(model, train_dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in tqdm(train_dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        weights = model(batch_x)
        loss = loss_fn(batch_y, weights, batch_x)  # batch_y is now [B, N, 252]
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_dataloader)

def validate(model, val_dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            weights = model(batch_x)
            loss = loss_fn(batch_y, weights, batch_x)  # batch_y is now [B, N, 252]
            total_loss += loss.item()
    
    return total_loss / len(val_dataloader)

def evaluate(model, test_dataloader, metrics, device):
    model.eval()
    all_weights = []
    all_returns = []
    all_batch_x = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            weights = model(batch_x)
            all_weights.append(weights)
            all_returns.append(batch_y)
            all_batch_x.append(batch_x)
    
    all_weights = torch.cat(all_weights, dim=0)  # [B, N]
    all_returns = torch.cat(all_returns, dim=0)  # [B, N, T]
    all_batch_x = torch.cat(all_batch_x, dim=0)  # [B, N, T]

    all_returns = torch.exp(all_returns) - 1
 
    # Calculate metrics using the PortfolioMetrics class
    results = {
        'Expected Return': metrics.expected_return(all_returns, all_weights).mean().item(),
        'Standard Deviation': metrics.standard_deviation(all_returns, all_weights, all_batch_x).mean().item(),
        'Sharpe Ratio': metrics.sharpe_ratio(all_returns, all_weights, all_batch_x).mean().item(),
        'Maximum Drawdown': metrics.maximum_drawdown(all_returns, all_weights).item(),
        'Sortino Ratio': metrics.sortino_ratio(all_returns, all_weights, all_batch_x).mean().item(),
        'Positive Return %': metrics.positive_return_percentage(all_returns, all_weights).item(),
        'Turnover': metrics.turnover(all_weights).item()
    }
    
    # Print detailed results
    print("\nDetailed Portfolio Performance Metrics:")
    print("-" * 40)
    for metric_name, value in results.items():
        print(f"{metric_name:20}: {value:.4f}")
    print("-" * 40)
    
    return results

def main():
    # Hyperparameters
    batch_size = 32
    hidden_dim = 128
    learning_rate = 1e-4
    num_epochs = 300
    risk_aversion = 1.0
    risk_free_rate = 0.0
    input_dim = 331
    
    # Load and prepare data
    dt, dt2, dt3 = load_data()
    train_dataloader, val_dataloader, test_dataloader = prepare_data(dt, batch_size=batch_size)
    
    # Initialize model and loss functions
    model = PortfolioModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    mv_loss = MeanVarianceLoss(risk_aversion=risk_aversion).to(device)
    sharpe_loss = MaxSharpeRatioLoss(risk_free_rate=risk_free_rate).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Train with Mean-Variance loss
        train_loss = train_epoch(model, train_dataloader, optimizer, sharpe_loss, device)
        val_loss = validate(model, val_dataloader, sharpe_loss, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
    
    # Evaluate best model
    metrics = PortfolioMetrics()
    results = evaluate(best_model, test_dataloader, metrics, device)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    main()