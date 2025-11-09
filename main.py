from data_loader import StockDataLoader
from data_preprocessor import DataPreprocessor
from model_lstm import LSTMModel, LSTMTrainer
from plot_utils import plot_forecast

# ---------- Parametry ----------
ticker = input("Podaj ticker spółki (np. TSLA, AAPL): ").strip().upper()
features = ["Open", "High", "Low", "Close", "Volume"]
seq_length = 500
hidden_size = 32
num_layers = 3
dropout = 0.2
epochs = 50
batch_size = 32
last_days = 100
n_steps = 30

# ---------- Dane ----------
loader = StockDataLoader(ticker)
data = loader.download_data()
preprocessor = DataPreprocessor(features=features, seq_length=seq_length)
X_train, X_test, y_train, y_test = preprocessor.prepare(data)

# ---------- Model ----------
input_size = X_train.shape[2]
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
trainer = LSTMTrainer(model)

# ---------- Trenowanie ----------
trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

# ---------- Ewaluacja ----------
trainer.evaluate(X_test, y_test)

# ---------- Wykres ----------
plot_forecast(data, preprocessor, model, X_test, last_days=last_days, n_steps=n_steps, features=features, device=trainer.device, ticker=ticker)
