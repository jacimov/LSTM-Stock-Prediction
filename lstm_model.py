"""LSTM model implementation for stock prediction."""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """LSTM-based model for stock price prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2
    ):
        """Initialize the LSTM model.

        Args:
            input_size: Size of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate between LSTM layers
        """
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size
        ).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get last output
        out = self.fc(out[:, -1, :])
        return out


class ModelTrainer:
    """Trainer class for LSTM model."""

    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(
        self, test_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate the model.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Tuple of (test loss, predictions, actual values)
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())

        return (
            total_loss / len(test_loader),
            np.array(predictions),
            np.array(actuals)
        )
