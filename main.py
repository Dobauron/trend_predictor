from data_loader import StockDataLoader
from data_preprocessor import DataPreprocessor
from model_lstm import LSTMModel, LSTMTrainer
from forecast_utils import multi_step_forecast
from plot_utils import plot_candlestick_forecast
import torch

# --- Parametry ---
ticker = input("Podaj ticker spółki (np. TSLA, AAPL, BTC-USD): ").strip().upper()
# Dodajemy kolumny z wskaźnikami
features = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MACD_signal", "MACD_hist"]
target_features = ["Open", "High", "Low", "Close", "Volume"]  # co model ma przewidywać
seq_length = 100
hidden_size = 64
num_layers = 2
dropout = 0.3
epochs = 50
batch_size = 32
last_days = 100
n_steps = 1

# --- Dane ---
loader = StockDataLoader(ticker)
data_raw = loader.download_data()                         # surowe OHLCV
data = loader.add_indicators_in_memory()                  # dodaj RSI i MACD

preprocessor = DataPreprocessor(features=features, target_features=target_features, seq_length=seq_length)
X_train, X_test, y_train, y_test = preprocessor.prepare(data)

# --- Model ---
input_size = X_train.shape[2]
model = LSTMModel(input_size=input_size, output_size=len(target_features),
                  hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
trainer = LSTMTrainer(model)

# --- Trening ---
trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
trainer.evaluate(X_test, y_test)

# --- Prognozy ---
preds_true = multi_step_forecast(model, X_test[-1], preprocessor, n_steps, device=trainer.device)
preds_close = preprocessor.target_scaler.inverse_transform(model(
    torch.tensor(X_test, dtype=torch.float32).to(trainer.device)
).detach().cpu().numpy())[:, target_features.index("Close")]

# --- Wykres ---
plot_candlestick_forecast(data, preds_true, preds_close, features, target_features, last_days, n_steps, ticker)

