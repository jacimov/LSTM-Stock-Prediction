import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from data_processor import DataProcessor
from lstm_model import LSTMPredictor, ModelTrainer
from typing import List, Tuple
import datetime

class AlphaDecayAnalyzer:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data_processor = DataProcessor(symbol, start_date, end_date)
        self.period_results = []
        
    def prepare_data_for_period(self, start_idx: int, end_idx: int, sequence_length: int = 60) -> Tuple:
        """Prepare data for a specific time period."""
        period_data = self.data_processor.data.iloc[start_idx:end_idx]
        prices = period_data['Adj Close'].values.reshape(-1, 1)
        scaled_data = self.data_processor.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i])
            
        X, y = np.array(X), np.array(y)
        return self.data_processor.split_data(X, y)
    
    def create_dataloaders(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray, 
                          batch_size: int = 32) -> Tuple:
        """Create PyTorch DataLoaders."""
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Create DataLoader objects
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        
        return train_loader, test_loader
    
    def analyze_period(self, start_date: str, end_date: str, 
                      hidden_size: int = 50, num_layers: int = 2,
                      epochs: int = 50) -> dict:
        """Analyze a specific time period."""
        # Get data for the period
        period_data = self.data_processor.data[start_date:end_date]
        start_idx = self.data_processor.data.index.get_loc(start_date)
        end_idx = self.data_processor.data.index.get_loc(end_date)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data_for_period(start_idx, end_idx)
        train_loader, test_loader = self.create_dataloaders(X_train, X_test, y_train, y_test)
        
        # Initialize and train model
        model = LSTMPredictor(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        trainer = ModelTrainer(model)
        
        # Training loop
        train_losses = []
        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_loader)
            train_losses.append(train_loss)
        
        # Evaluate
        test_loss, predictions, actuals = trainer.evaluate(test_loader)
        
        # Convert back to original scale
        predictions = self.data_processor.inverse_transform(predictions)
        actuals = self.data_processor.inverse_transform(actuals)
        
        # Calculate metrics
        metrics = self.data_processor.calculate_metrics(actuals, predictions)
        
        return {
            'period': (start_date, end_date),
            'metrics': metrics,
            'predictions': predictions,
            'actuals': actuals,
            'train_losses': train_losses
        }
    
    def analyze_alpha_decay(self, period_length: str = '5Y') -> List[dict]:
        """Analyze alpha decay across multiple time periods."""
        # Fetch data
        self.data_processor.fetch_data()
        
        # Split data into periods
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        current_date = start_date
        
        while current_date < end_date:
            period_end = min(current_date + pd.Timedelta(period_length), end_date)
            
            # Analyze period
            results = self.analyze_period(
                str(current_date.date()),
                str(period_end.date())
            )
            self.period_results.append(results)
            
            current_date = period_end
            
        return self.period_results
    
    def plot_results(self):
        """Plot analysis results."""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot R-squared values over time
        periods = [pd.to_datetime(result['period'][0]) for result in self.period_results]
        r2_values = [result['metrics']['R2'] for result in self.period_results]
        
        ax1.plot(periods, r2_values, marker='o')
        ax1.set_title('Alpha Decay Analysis: R-squared over Time')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('R-squared')
        ax1.grid(True)
        
        # Plot RMSE values over time
        rmse_values = [result['metrics']['RMSE'] for result in self.period_results]
        ax2.plot(periods, rmse_values, marker='o', color='red')
        ax2.set_title('Model Error over Time')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('RMSE')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('alpha_decay_analysis.png')
        plt.close()

if __name__ == "__main__":
    # Example usage
    analyzer = AlphaDecayAnalyzer(
        symbol='^GSPC',  # S&P 500
        start_date='1960-01-01',
        end_date='2023-01-01'
    )
    
    # Analyze alpha decay across periods
    results = analyzer.analyze_alpha_decay()
    
    # Plot results
    analyzer.plot_results()