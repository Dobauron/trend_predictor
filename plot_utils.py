# plot_utils.py
import matplotlib.pyplot as plt
import torch
import numpy as np
from datetime import timedelta

def plot_forecast(data, preprocessor, model, X_test, last_days=100, n_steps=5, features=None, device='cpu', ticker='TICKER'):
    model.eval()
    with torch.no_grad():
        # Predykcje na X_test
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds_scaled = model(X_test_tensor).cpu().numpy().flatten()
        preds_true = preprocessor.target_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

    # Multi-step forecast
    multi_preds = []
    last_seq = X_test[-1]
    current_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

    for step in range(n_steps):
        pred = model(current_seq)
        multi_preds.append(pred.item())

        # sliding window: usuń pierwszy krok, dodaj nowy
        new_step = current_seq[:, 1:, :].clone()
        next_features = current_seq[:, -1, :].clone()

        # wstaw predykcję w kolumnę Close i reszta cech taka sama jak ostatnia
        pred_scaled = preprocessor.target_scaler.transform(np.array([[pred.item()]])).item()
        next_features[0, features.index("Close")] = pred_scaled

        current_seq = torch.cat((new_step, next_features.unsqueeze(1)), dim=1)

    multi_preds_true = preprocessor.target_scaler.inverse_transform(np.array(multi_preds).reshape(-1,1)).flatten()

    # Daty
    dates_hist = data.index[-last_days:]
    last_date = dates_hist[-1]
    dates_forecast = [last_date + timedelta(days=i+1) for i in range(n_steps)]

    # Wykres
    plt.figure(figsize=(12,6))
    # ostatnie ceny historyczne
    plt.plot(dates_hist, data['Close'].iloc[-last_days:], label="Ostatnie ceny", linewidth=2)
    # predykcje na test set
    plt.plot(dates_hist[-len(preds_true):], preds_true[-len(dates_hist):], linestyle='--', label="Predykcja LSTM (historyczne)")
    # multi-step forecast
    plt.plot(dates_forecast, multi_preds_true, linestyle='--', color='red', label=f"Prognoza {n_steps}-dniowa")
    plt.title(f"📈 Prognoza LSTM dla {ticker}")
    plt.xlabel("Data")
    plt.ylabel("Cena [USD]")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()
