"""
Testy dla plot_utils.py
Sprawdza:
- czy można utworzyć obiekt CandlestickPlotter
- czy metoda prepare_forecast_data działa i zwraca poprawne wymiary
"""

import pandas as pd
import numpy as np
from plot_utils import CandlestickPlotter
from data_preprocessor import DataPreprocessor
from model_lstm import LSTMModel

def test_plotter_prepare():
    n_samples = 50
    features = ["Open","High","Low","Close","Volume"]
    df = pd.DataFrame(np.random.rand(n_samples, len(features))*100, columns=features)
    df.index = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")  # 🟢 dodaj to

    preprocessor = DataPreprocessor(features=features, target_features=["Close"], seq_length=5)
    X_train, X_test, y_train, y_test = preprocessor.prepare(df)

    model = LSTMModel(input_size=len(features), output_size=len(features))
    plotter = CandlestickPlotter(df, features, preprocessor, model, X_test)

    forecast_df, preds_close = plotter.prepare_forecast_data(n_steps=3)

    assert forecast_df is not None
    assert preds_close is not None

