import torch
import numpy as np
import pandas as pd
from indicators import compute_rsi, compute_macd


def multi_step_forecast(model, last_sequence, preprocessor, n_steps, device='cpu'):
    """
    Generuje prognozy wielu dni do przodu, uwzględniając pełną sekwencję 9 cech
    i dynamiczne wyliczanie RSI/MACD dla przewidywanych dni.

    model          : wytrenowany model LSTM
    last_sequence  : ostatnia sekwencja testowa (np. X_test[-1]), shape (seq_length, 9)
    preprocessor   : obiekt DataPreprocessor (ze scalerami)
    n_steps        : ile dni prognozować do przodu
    device         : 'cpu' lub 'cuda'
    """
    model.eval()
    preds_true = []

    # kopiujemy ostatnią sekwencję do tensoru
    current_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    # tworzymy listę cen zamknięcia, potrzebną do RSI/MACD
    close_series = list(last_sequence[:, 3])  # kolumna Close

    for step in range(n_steps):
        with torch.no_grad():
            # predykcja w przestrzeni skalowanej
            pred_scaled = model(current_seq).detach().cpu().numpy().flatten()

        # odwrócenie skalowania do rzeczywistych wartości 5 przewidywanych cech
        pred_true_5 = preprocessor.target_scaler.inverse_transform(pred_scaled.reshape(1, -1)).flatten()

        # tworzymy pełną tablicę 9 cech
        new_day = np.zeros(last_sequence.shape[1])
        new_day[:5] = pred_true_5  # Open, High, Low, Close, Volume

        # dodajemy Close do listy historycznej
        close_series.append(new_day[3])

        # obliczamy RSI i MACD na podstawie aktualnej historii Close
        rsi_series = compute_rsi(pd.Series(close_series))
        macd, macd_signal, macd_hist = compute_macd(pd.Series(close_series))
        new_day[5] = rsi_series.iloc[-1]
        new_day[6] = macd.iloc[-1]
        new_day[7] = macd_signal.iloc[-1]
        new_day[8] = macd_hist.iloc[-1]

        preds_true.append(new_day)

        # przygotowujemy nową sekwencję do kolejnego kroku
        new_step = current_seq[:, 1:, :].clone()
        next_features = torch.tensor(new_day, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
        current_seq = torch.cat((new_step, next_features), dim=1)

    preds_true = np.array(preds_true)
    return preds_true
