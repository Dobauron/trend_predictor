import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LSTMTrainer:
    def __init__(self, model, lr=0.001, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train, epochs=15, batch_size=32):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss/len(loader):.4f}")

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)
            preds = self.model(X_test_tensor).cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()

        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)
        print("\n✅ Wyniki regresji:")
        print(f"📉 MAE: {mae:.6f}")
        print(f"📊 RMSE: {rmse:.6f}")
        print(f"📈 R²: {r2:.4f}")
        return mae, rmse, r2
