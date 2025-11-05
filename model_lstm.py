import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import StockDataLoader
from data_preprocessor import DataPreprocessor


# ===================== MODEL LSTM (PyTorch) =====================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # ostatni krok sekwencji
        out = self.fc(out)
        return self.sigmoid(out)


# ===================== TRENER =====================
class LSTMTrainer:
    def __init__(self, model, lr=0.001, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train, epochs=15, batch_size=32):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss / len(loader):.4f}")

    def evaluate(self, X_test, y_test):
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            preds = self.model(X_test)
            preds = (preds > 0.5).float()

        acc = accuracy_score(y_test.cpu(), preds.cpu())
        print(f"\n✅ Dokładność: {acc:.3f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test.cpu(), preds.cpu(), zero_division=0))
        print("Macierz pomyłek:")
        print(confusion_matrix(y_test.cpu(), preds.cpu()))


# ===================== TEST CAŁEGO PIPELINE’U =====================
if __name__ == "__main__":
    ticker = input("Podaj ticker spółki (np. TSLA, AAPL, BTC-USD): ").strip().upper()

    # 1️⃣ Wczytanie danych z CSV (jeśli brak — pobierze z Yahoo)
    loader = StockDataLoader(ticker)
    data = loader.download_data()
    data = loader.add_indicators()

    # 2️⃣ Przygotowanie danych (sekwencje + skalowanie)
    features = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MACD_signal", "MACD_hist"]
    preprocessor = DataPreprocessor(features=features, seq_length=180)
    X_train, X_test, y_train, y_test = preprocessor.prepare(data)

    # 3️⃣ Model LSTM (PyTorch)
    input_size = X_train.shape[2]

    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3)

    # 4️⃣ Trenowanie
    trainer = LSTMTrainer(model)
    trainer.train(X_train, y_train, epochs=200, batch_size=32)

    # 5️⃣ Ewaluacja
    trainer.evaluate(X_test, y_test)
