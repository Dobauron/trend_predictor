import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=3, dropout=0.25):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class LSTMTrainer:
    def __init__(self, model, lr=0.001, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train),
                                             batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
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
        print(f"\nMAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.4f}")
        return mae, rmse, r2
