"""Data processing utilities for stock prediction."""
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    """Data processor for stock market data."""

    def __init__(self, symbol: str, start_date: str, end_date: str):
        """Initialize the data processor.

        Args:
            symbol: Stock symbol to process
            start_date: Start date for data collection
            end_date: End date for data collection
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance.

        Returns:
            DataFrame containing stock data
        """
        stock = yf.Ticker(self.symbol)
        self.data = stock.history(
            start=self.start_date,
            end=self.end_date
        )
        print("Available columns:", self.data.columns)  # Debug print
        return self.data

    def prepare_data(self, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model."""
        prices = self.data['Close'].values.reshape(-1, 1)  # Changed from 'Adj Close' to 'Close'
        scaled_data = self.scaler.fit_transform(prices)
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i])
        return np.array(X), np.array(y)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets.

        Args:
            X: Features array
            y: Target array
            test_size: Proportion of data to use for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale.

        Args:
            data: Scaled data array

        Returns:
            Data array in original scale
        """
        return self.scaler.inverse_transform(data)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> dict:
        """Calculate performance metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary containing performance metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
