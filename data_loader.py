import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time

class StockDataLoader:
    def __init__(self, ticker: str, start: str = "2022-01-01",
                 end=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                 interval="1d", data_folder="data"):
        self.ticker = ticker.upper()
        self.start = start
        self.end = end
        self.interval = interval
        self.data_folder = data_folder
        self.data = None
        os.makedirs(self.data_folder, exist_ok=True)

    def download_data(self, max_retries=3):
        csv_path = os.path.join(self.data_folder, f"{self.ticker}_raw.csv")

        if os.path.exists(csv_path):
            print(f"📄 Wczytuję dane z CSV: {csv_path}")
            self.data = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            if not self.data.empty:
                return self.data

        print(f"🌐 Pobieram dane dla: {self.ticker} ({self.start} → {self.end})")
        for attempt in range(max_retries):
            try:
                data = yf.download(self.ticker, start=self.start, end=self.end,
                                   interval=self.interval, progress=False, timeout=60)
                if data.empty:
                    time.sleep(2)
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

                data.to_csv(csv_path)
                self.data = data
                return self.data

            except Exception as e:
                print(f"❌ Błąd pobierania (próba {attempt+1}): {e}")
                time.sleep(2)

        raise ValueError(f"Nie udało się pobrać danych dla {self.ticker}")

    @staticmethod
    def compute_rsi(series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
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

    def add_indicators_in_memory(self):
        if self.data is None or self.data.empty:
            raise ValueError("Brak danych. Najpierw pobierz dane metodą download_data().")
        df = self.data.copy()
        df["RSI"] = self.compute_rsi(df["Close"])
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = self.compute_macd(df["Close"])
        df.dropna(inplace=True)
        return df
