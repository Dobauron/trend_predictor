# ==========================================
# 📈 OOP LSTM – Klasyfikacja trendu z RSI i MACD
# ==========================================

# ============================================================
# 🔹 Klasa 1: Pobieranie danych i obliczanie wskaźników
# ============================================================
import yfinance as yf


class StockDataLoader:
    def __init__(self, ticker: str, start: str = "2015-01-01", end: str = "2025-01-01"):
        self.ticker = ticker.upper()
        self.start = start
        self.end = end
        self.data = None

    def download_data(self):
        print(f"Pobieram dane dla: {self.ticker} ...")
        data = yf.download(self.ticker, start=self.start, end=self.end)
        if data.empty:
            raise ValueError("❌ Brak danych. Sprawdź ticker.")
        self.data = data
        return self.data

    @staticmethod
    def compute_rsi(series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def compute_macd(series, short=12, long=26, signal=9):
        short_ema = series.ewm(span=short, adjust=False).mean()
        long_ema = series.ewm(span=long, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist

    def add_indicators(self):
        self.data["RSI"] = self.compute_rsi(self.data["Close"])
        self.data["MACD"], self.data["MACD_signal"], self.data["MACD_hist"] = self.compute_macd(self.data["Close"])
        self.data.dropna(inplace=True)
        return self.data
