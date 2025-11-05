from data_loader import StockDataLoader
from data_preprocessor import DataPreprocessor
from model_lstm import LSTMModel
from evaluator import ModelEvaluator

class StockTrendPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.features = ["Close", "RSI", "MACD", "MACD_signal", "MACD_hist"]
        self.seq_length = 30

    def run(self):
        # Dane
        loader = StockDataLoader(self.ticker)
        data = loader.download_data()
        # data = loader.add_indicators()

        # Przygotowanie danych
        preprocessor = DataPreprocessor(self.features, self.seq_length)
        X_train, X_test, y_train, y_test = preprocessor.prepare(data)

        # Model
        model = LSTMModel(input_shape=(self.seq_length, len(self.features)))
        history = model.train(X_train, y_train)

        # Ewaluacja
        y_pred = model.predict(X_test)
        evaluator = ModelEvaluator()
        evaluator.evaluate(y_test, y_pred)
        evaluator.plot_training(history)
        evaluator.plot_trends(data, y_test, y_pred, self.ticker)


if __name__ == "__main__":
    ticker = input("Podaj ticker spółki (np. AAPL, TSLA, BTC-USD): ").strip().upper()
    predictor = StockTrendPredictor(ticker)
    predictor.run()
