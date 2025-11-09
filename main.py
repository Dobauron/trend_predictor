from data_loader import StockDataLoader
from data_preprocessor import DataPreprocessor
from model_lstm import LSTMModel, LSTMTrainer
from plot_utils import plot_candlestick_forecast

# ---------- Parametry ----------
ticker = input("Podaj ticker spółki (np. TSLA, AAPL): ").strip().upper()
features = ["Open", "High", "Low", "Close", "Volume"]
target_features = ["Open", "High", "Low", "Close", "Volume"]
seq_length = 600 # z ilu dni model bierze dane
hidden_size = 64 #  ilosc neuronow w warstwie ukrytej
num_layers = 3 # ilosc warstw ukrytych
dropout = 0.2 # procent odrzuconych neuronow
epochs = 15 # ilosc iteracji
batch_size = 32 # ilość danych w pojedynczej iteracji
last_days = 100 # ilosc dni przedstawionych na wykresie wstecz
n_steps = 5 #ilosc dni do przewidzenia

# ---------- Dane ----------
print("\n📊 Konfiguracja modelu:")
print(f"Ticker: {ticker}")
print(f"Sekwencja: {seq_length}, Hidden: {hidden_size}, Layers: {num_layers}, Dropout: {dropout}")
print(f"Epoki: {epochs}, Batch: {batch_size}, Wykres: {last_days} dni\n")

# ---------- Dane ----------
loader = StockDataLoader(ticker)
data = loader.download_data()
preprocessor = DataPreprocessor(features=features, target_features=target_features, seq_length=seq_length)
X_train, X_test, y_train, y_test = preprocessor.prepare(data)

# ---------- Model ----------
input_size = X_train.shape[2]
model = LSTMModel(input_size=input_size, output_size=5, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
trainer = LSTMTrainer(model)

# ---------- Trenowanie ----------
trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

# ---------- Ewaluacja ----------
trainer.evaluate(X_test, y_test)

# ---------- Wykres i prognoza ----------
plot_candlestick_forecast(
    data,
    preprocessor,
    model,
    X_test,
    last_days=last_days,
    n_steps=n_steps,
    features=features,
    device=trainer.device,
    ticker=ticker,
)