# End-to-End Portfolio Optimization

This project implements an end-to-end deep learning approach for portfolio optimization using PyTorch. The model learns to optimize portfolio weights directly from return series data, providing a data-driven alternative to traditional portfolio optimization methods.

## Features

- Deep learning-based portfolio optimization
- End-to-end training with custom loss functions
- Support for multiple evaluation metrics
- Flexible architecture for different asset classes
- Real-time portfolio rebalancing
- Risk-adjusted return optimization

## Project Structure

### Core Modules

1. **Score Block**
   - Input: Return series [B, N, T] (Batch, Number of assets, Time steps)
   - Output: Asset scores [B, N]
   - Architecture: LSTM-based neural network
   - Purpose: Learns to score assets based on their historical returns

2. **Portfolio Block**
   - Input: Asset scores [B, N]
   - Output: Portfolio weights [B, N]
   - Function: Converts scores to valid portfolio weights using softmax
   - Constraint: Weights sum to 1 (fully invested portfolio)

3. **Loss Functions**
   - Mean-Variance Portfolio Loss
     - Optimizes risk-adjusted returns
     - Balances expected return and portfolio variance
     - Customizable risk aversion parameter
   - Maximum Sharpe Ratio Loss
     - Maximizes risk-adjusted returns
     - Optimizes the Sharpe ratio directly

### Evaluation Metrics

1. Expected Return (E(R))
   - Annualized portfolio return
   - Calculated from weighted asset returns

2. Standard Deviation (Std(R))
   - Portfolio risk measure
   - Annualized volatility

3. Sharpe Ratio
   - Risk-adjusted return metric
   - Excess return per unit of risk

4. Sortino Ratio
   - Downside risk-adjusted return
   - Considers only negative returns

5. Maximum Drawdown (MDD)
   - Largest peak-to-trough decline
   - Worst historical loss

6. Positive Return Percentage
   - Percentage of positive returns
   - Win rate metric

7. Frobenius Norm
   - Matrix norm for covariance
   - Risk structure measure

8. Turnover
   - Portfolio rebalancing cost
   - Measures weight changes

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- NumPy 1.21.0+
- Pandas 1.3.0+
- Matplotlib 3.4.0+
- scikit-learn 0.24.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/end2end-portfolio-optimization.git
cd end2end-portfolio-optimization
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Run the main script:
```bash
python main.py
```

2. For training:
```bash
python train.py
```

### Advanced Usage

1. Customize model parameters:
```python
from portfolio_block import PortfolioBlock
from score_block import ScoreBlock

# Initialize blocks with custom parameters
score_block = ScoreBlock(input_dim=10, hidden_dim=64)
portfolio_block = PortfolioBlock(num_assets=10)
```

2. Use custom loss functions:
```python
from loss import MeanVarianceLoss, SharpeRatioLoss

# Choose loss function
loss_fn = MeanVarianceLoss(risk_aversion=0.5)
# or
loss_fn = SharpeRatioLoss(risk_free_rate=0.02)
```

## Data

The project uses S&P 500 price data for demonstration. The data includes:
- Daily closing prices
- High prices
- Low prices

Data preprocessing includes:
- Log returns calculation
- Normalization
- Time series windowing

## Model Architecture

The model combines:
1. LSTM-based feature extraction
2. Portfolio weight optimization
3. End-to-end training with custom loss functions

## Training

The training process:
1. Uses historical return windows
2. Optimizes portfolio weights
3. Evaluates performance metrics
4. Saves training history

## License

MIT License# R20_End2End-Portfolio-Optimization
# P05_End2End-Portfolio-Optimization
