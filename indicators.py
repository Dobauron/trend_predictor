# indicators.py
import pandas as pd

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Oblicza RSI (Relative Strength Index) dla podanej serii cen zamknięcia."""
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series, short: int = 12, long: int = 26, signal: int = 9):
    """Oblicza MACD, linię sygnału i histogram dla podanej serii cen zamknięcia."""
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist
