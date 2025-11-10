import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from indicators import compute_rsi, compute_macd


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
        """Pobiera dane surowe tylko jeśli plik CSV nie istnieje"""
        csv_path = os.path.join(self.data_folder, f"{self.ticker}_raw.csv")

        if os.path.exists(csv_path):
            print(f"📄 Wczytuję dane z istniejącego pliku CSV: {csv_path}")
            self.data = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            if not self.data.empty:
                print(f"✅ Wczytano dane z pliku: {len(self.data)} rekordów.")
                return self.data

        # Pobieranie z Yahoo Finance
        print(f"🌐 Pobieram dane dla: {self.ticker} ({self.start} → {self.end})")
        for attempt in range(max_retries):
            try:
                data = yf.download(self.ticker, start=self.start, end=self.end,
                                   interval=self.interval, progress=False, timeout=60)
                if data.empty:
                    print(f"⚠️ Próba {attempt + 1}/{max_retries}: brak danych.")
                    time.sleep(2)
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

                # Zapisz surowe dane do CSV
                data.to_csv(csv_path)
                print(f"💾 Dane zapisane do: {csv_path} ({len(data)} rekordów)")
                self.data = data
                return self.data

            except Exception as e:
                print(f"❌ Błąd pobierania (próba {attempt + 1}): {e}")
                time.sleep(2)

        raise ValueError(f"❌ Nie udało się pobrać danych dla {self.ticker} po {max_retries} próbach.")

    def add_indicators_in_memory(self):
        """Dodaje RSI i MACD w pamięci, bez zmiany CSV"""
        if self.data is None or self.data.empty:
            raise ValueError("❌ Brak danych. Najpierw pobierz dane metodą download_data().")

        df = self.data.copy()
        df["RSI"] = compute_rsi(df["Close"])
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["Close"])
        df.dropna(inplace=True)
        return df


if __name__ == "__main__":
    ticker = input("Podaj ticker (np. TSLA, AAPL, BTC-USD): ").strip().upper()
    loader = StockDataLoader(ticker)

    # Pobranie surowych danych (tylko jeśli plik nie istnieje)
    data = loader.download_data()
    print(f"\n✅ Surowe dane pobrane lub wczytane: {len(data)} rekordów")

    # Dodanie wskaźników w pamięci
    data_with_indicators = loader.add_indicators_in_memory()
    print(f"\n📊 Dane z wskaźnikami (pamięć): {len(data_with_indicators)} rekordów")
    print(data_with_indicators.head())
