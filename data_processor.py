import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

class DataProcessor:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler()
        self.data = None
        self.scaled_data = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        return self.data
    
    def prepare_data(self, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model."""
        # Use adjusted close prices
        prices = self.data['Adj Close'].values.reshape(-1, 1)
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i-sequence_length:i])
            y.append(self.scaled_data[i])
            
        return np.array(X), np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple:
        """Split data into training and testing sets."""
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Convert scaled values back to original scale."""
        return self.scaler.inverse_transform(scaled_data)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate performance metrics."""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        nrmse = rmse / (np.max(y_true) - np.min(y_true))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        return {
            'RMSE': rmse,
            'NRMSE': nrmse,
            'R2': r2
        }
