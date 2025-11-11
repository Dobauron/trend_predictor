import pytest
import numpy as np
import pandas as pd
import torch
from data_preprocessor import DataPreprocessor
from model_lstm import LSTMModel

@pytest.mark.parametrize("n_samples", [50, 100])
def test_rsi_macd_used_in_training(n_samples):
    """
    Test sprawdza, czy RSI i MACD są dodawane do danych wejściowych
    i czy model LSTM dostaje tensor z tymi cechami.
    """

    # Tworzymy sztuczne dane
    features = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MACD_signal", "MACD_hist"]
    target_features = ["Open", "High", "Low", "Close", "Volume"]
    df = pd.DataFrame(np.random.rand(n_samples, len(features))*100, columns=features)

    # Inicjalizacja preprocesora
    seq_length = 5
    preprocessor = DataPreprocessor(features=features, target_features=target_features, seq_length=seq_length)
    X_train, X_test, y_train, y_test = preprocessor.prepare(df)

    # Sprawdzenie wymiarów tensorów
    # X_train powinno mieć shape: (num_samples, seq_length, n_features)
    assert X_train.shape[2] == len(features), "Tensor wejściowy nie zawiera wszystkich cech, w tym RSI i MACD"

    # Sprawdzenie, czy LSTM dostaje odpowiedni input_size
    model = LSTMModel(input_size=X_train.shape[2], output_size=len(target_features))
    sample_tensor = torch.tensor(X_train[:2], dtype=torch.float32)
    output = model(sample_tensor)

    # Output powinien mieć shape: (batch, output_size)
    assert output.shape[1] == len(target_features), "Model nie przewiduje wszystkich target_features"

    # Sprawdzenie, czy wartości RSI/MACD są faktycznie w tensorze (czy nie same zera)
    rsi_idx = features.index("RSI")
    macd_idx = features.index("MACD")
    assert np.any(X_train[:, :, rsi_idx] != 0), "RSI w tensorze jest zerowe"
    assert np.any(X_train[:, :, macd_idx] != 0), "MACD w tensorze jest zerowe"
