"""
Testy dla data_preprocessor.py
Sprawdza:
- czy sekwencje X, y są poprawnie tworzone
- czy długość sekwencji odpowiada seq_length
- czy rozmiary danych treningowych i testowych są zgodne
"""

import pandas as pd
import numpy as np
from data_preprocessor import DataPreprocessor

def test_create_sequences():
    # sztuczne dane
    df = pd.DataFrame({
        "Open": np.arange(10),
        "High": np.arange(10),
        "Low": np.arange(10),
        "Close": np.arange(10),
        "Volume": np.arange(10)
    })
    seq_length = 3
    preprocessor = DataPreprocessor(features=["Open","High","Low","Close","Volume"],
                                    target_features=["Close"], seq_length=seq_length)
    X_train, X_test, y_train, y_test = preprocessor.prepare(df)
    assert X_train.shape[1] == seq_length, "Długość sekwencji X_train nie zgadza się"
    assert X_train.shape[0] + X_test.shape[0] == len(df) - seq_length, "Liczba sekwencji nie zgadza się"
