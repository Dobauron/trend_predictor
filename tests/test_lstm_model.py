"""
Testy dla model_lstm.py
Sprawdza:
- czy model LSTM można utworzyć i wykonać forward pass
- czy wyjście ma odpowiedni kształt
"""

import torch
from model_lstm import LSTMModel

def test_lstm_forward():
    input_size = 5
    output_size = 1
    seq_length = 10
    batch_size = 2
    model = LSTMModel(input_size=input_size, output_size=output_size)
    x = torch.rand(batch_size, seq_length, input_size)
    y = model(x)
    assert y.shape == (batch_size, output_size), "Kształt wyjścia LSTM nie zgadza się"
