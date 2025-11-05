# ==========================================
# 📈 OOP LSTM – Klasyfikacja trendu z RSI i MACD
# ==========================================

# ============================================================
# 🔹 Klasa 1: Pobieranie danych i obliczanie wskaźników
# ============================================================
import yfinance as yf
import os
import pandas as pd
import time
from datetime import datetime, timedelta

class StockDataLoader:
    def __init__(self, ticker: str, start: str = "2015-01-01", end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), data_folder="data"):
        self.ticker = ticker.upper()
        self.start = start
        self.end = end
        self.data_folder = data_folder
        self.data = None
        os.makedirs(self.data_folder, exist_ok=True)  # tworzy folder jeśli nie istnieje

    def download_data(self, max_retries=3):
        csv_path = os.path.join(self.data_folder, f"{self.ticker}.csv")

        # Jeśli CSV istnieje → wczytaj
        if os.path.exists(csv_path):
            print(f"📄 Wczytuję dane z pliku CSV: {csv_path}")
            self.data = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            if not self.data.empty:
                print(f"✅ Wczytano dane z pliku: {len(self.data)} rekordów.")
                return self.data
            else:
                print("⚠️ Plik CSV istnieje, ale jest pusty — spróbuję pobrać ponownie.")

        # Pobieranie z Yahoo Finance (z retry)
        print(f"🌐 Pobieram dane dla: {self.ticker} ({self.start} → {self.end})")
        for attempt in range(max_retries):
            try:
                data = yf.download(self.ticker, start=self.start, end=self.end, progress=False, timeout=60)
                if data.empty:
                    print(f"⚠️ Próba {attempt+1}/{max_retries}: brak danych (data.empty=True).")
                    time.sleep(2)
                    continue

                # Spłaszcz kolumny, jeśli mają MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

                # Zapisz tylko jeśli dane istnieją
                data.to_csv(csv_path)
                print(f"💾 Dane zapisane do: {csv_path} ({len(data)} rekordów)")
                self.data = data
                return self.data

            except Exception as e:
                print(f"❌ Błąd pobierania (próba {attempt+1}): {e}")
                time.sleep(2)

        raise ValueError(f"❌ Nie udało się pobrać danych dla {self.ticker} po {max_retries} próbach.")

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
        """Dodaje RSI i MACD do danych i zapisuje do CSV."""
        if self.data is None or self.data.empty:
            raise ValueError("❌ Brak danych. Najpierw pobierz dane metodą download_data().")

        print("📊 Dodaję wskaźniki RSI i MACD...")

        self.data["RSI"] = self.compute_rsi(self.data["Close"])
        self.data["MACD"], self.data["MACD_signal"], self.data["MACD_hist"] = self.compute_macd(self.data["Close"])
        self.data.dropna(inplace=True)

        # Zapisz zaktualizowane dane do CSV
        csv_path = os.path.join(self.data_folder, f"{self.ticker}.csv")
        self.data.to_csv(csv_path)
        print(f"💾 Zaktualizowane dane zapisane do: {csv_path}")

        return self.data


if __name__ == "__main__":
    print("=== TEST: Data Loader ===")
    ticker = input("Podaj ticker (np. TSLA, AAPL, BTC-USD): ").strip().upper()

    loader = StockDataLoader(ticker)
    data = loader.download_data()
    print(f"\n✅ Dane pobrane lub wczytane. Liczba rekordów: {len(data)}")

    # Dodaj RSI i MACD
    data = loader.add_indicators()
    print("\n📊 Podgląd danych z RSI i MACD:")
    print(data.head())

    # Sprawdź czy plik istnieje
    import os
    csv_path = os.path.join(loader.data_folder, f"{ticker}.csv")
    print(f"\n📁 Plik CSV istnieje: {os.path.exists(csv_path)} ({csv_path})")