# Alpha Decay Analysis in Stock Market Prediction

This project implements an LSTM-based approach to analyze potential alpha decay in stock market prediction strategies. It examines over 60 years of historical data from major indices and evaluates the model's predictive accuracy across different time periods.

## Features

- LSTM neural network implementation using PyTorch
- Comprehensive data pipeline for time series processing
- Feature engineering across different market regimes
- Statistical analysis using RMSE, NRMSE, and R-squared metrics
- Visualization of predicted vs actual stock prices
- Analysis of alpha decay trends over time

## Project Structure

- `data_processor.py`: Handles data fetching, preprocessing, and feature engineering
- `lstm_model.py`: Implements the LSTM model architecture and training logic
- `alpha_decay_analysis.py`: Main script for analyzing alpha decay across time periods
- `requirements.txt`: Project dependencies

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Basic usage with default parameters:
```python
from alpha_decay_analysis import AlphaDecayAnalyzer

analyzer = AlphaDecayAnalyzer(
    symbol='^GSPC',  # S&P 500
    start_date='1960-01-01',
    end_date='2023-01-01'
)

# Analyze alpha decay across periods
results = analyzer.analyze_alpha_decay()

# Plot results
analyzer.plot_results()
```

2. Customize analysis parameters:
```python
# Analyze with custom period length
results = analyzer.analyze_alpha_decay(period_length='3Y')

# Analyze specific period with custom model parameters
period_results = analyzer.analyze_period(
    start_date='2010-01-01',
    end_date='2015-01-01',
    hidden_size=100,
    num_layers=3,
    epochs=100
)
```

## Output

The analysis generates:
1. Performance metrics (RMSE, NRMSE, R-squared) for each time period
2. Visualization of alpha decay trends over time
3. Comparison plots of predicted vs actual stock prices
4. Training loss curves for model evaluation

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- yfinance
- matplotlib
- seaborn

## Notes

- The analysis uses adjusted closing prices from Yahoo Finance
- Default sequence length for LSTM input is 60 days
- Models are trained on 80% of data and tested on 20%
- GPU acceleration is automatically used if available
