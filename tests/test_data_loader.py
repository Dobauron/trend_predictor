"""
Testy dla data_loader.py
Sprawdza:
- czy dane można pobrać z Yahoo Finance
- czy wskaźniki RSI i MACD są poprawnie dodawane
- czy struktura DataFrame jest poprawna
"""

from data_loader import StockDataLoader

def test_download_data():
    loader = StockDataLoader("BTC-USD", start="2023-01-01", end="2023-02-01")
    df = loader.download_data()
    assert not df.empty, "DataFrame nie może być pusty"
    assert all(col in df.columns for col in ["Open","High","Low","Close","Volume"]), "Brakuje kolumn"

def test_add_indicators():
    loader = StockDataLoader("BTC-USD", start="2023-01-01", end="2023-02-01")
    df = loader.download_data()
    df_ind = loader.add_indicators_in_memory()
    assert "RSI" in df_ind.columns, "Brak kolumny RSI"
    assert "MACD" in df_ind.columns, "Brak kolumny MACD"
    assert len(df_ind) < len(df) or len(df_ind) == len(df), "Długość po dropna powinna być <= oryginału"
