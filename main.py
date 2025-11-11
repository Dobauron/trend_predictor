from data_loader import StockDataLoader
from data_preprocessor import DataPreprocessor
from model_lstm import LSTMModel, LSTMTrainer
from plot_utils import CandlestickPlotter

# ---------- Parametry ----------
ticker = input("Podaj ticker spółki (np. TSLA, AAPL, BTC-USD): ").strip().upper()
seq_length = 60
hidden_size = 128
num_layers = 4
dropout = 0.25
epochs = 100
batch_size = 32
last_days = 100
n_steps = 1

# ---------- Dane ----------
loader = StockDataLoader(ticker)
data = loader.download_data()
data = loader.add_indicators_in_memory()  # Dodaje RSI i MACD

# Dodanie wskaźników do cech
features = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MACD_signal", "MACD_hist"]
target_features = ["Open", "High", "Low", "Close", "Volume"]

preprocessor = DataPreprocessor(features=features, target_features=target_features, seq_length=seq_length)
X_train, X_test, y_train, y_test = preprocessor.prepare(data)

# ---------- Model ----------
input_size = X_train.shape[2]
model = LSTMModel(input_size=input_size, output_size=len(target_features),
                  hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
trainer = LSTMTrainer(model)

# ---------- Trenowanie ----------
trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

# ---------- Ewaluacja ----------
trainer.evaluate(X_test, y_test)

# ---------- Wykres ----------
plotter = CandlestickPlotter(data, features, preprocessor, model, X_test)
forecast_df, preds_close = plotter.prepare_forecast_data(n_steps=n_steps)
plotter.plot_forecast(forecast_df, preds_close, last_days=last_days, ticker=ticker)
