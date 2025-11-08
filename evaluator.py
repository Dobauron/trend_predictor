import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)  # bez sigmoid, bo to regresja

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # bierzemy ostatni krok sekwencji
        out = self.fc(out)
        return out


class LSTMTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.MSELoss()           # regresja → MSE
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, X_train, y_train, epochs=50, batch_size=32, device="cpu"):
        self.model.to(device)
        self.model.train()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss / len(loader):.6f}")

    def evaluate(self, X_test, y_test, device="cpu"):
        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()

        # metryki regresji
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        r2 = r2_score(y_true, predictions)

        print("\n✅ Wyniki regresji:")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.4f}")

        return predictions, y_true
