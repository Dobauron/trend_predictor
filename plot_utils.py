import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle

def plot_candlestick_forecast(data, preprocessor, model, X_test, features,
                               last_days=100, n_steps=5, device='cpu', ticker='TICKER'):
    """
    data        : DataFrame z pełnymi danymi historycznymi
    preprocessor: obiekt DataPreprocessor
    model       : wytrenowany model LSTM
    X_test      : dane testowe (sekwencje)
    features    : lista cech ['Open','High','Low','Close','Volume']
    last_days   : ile ostatnich dni historycznych pokazać
    n_steps     : ile dni prognozować do przodu
    """
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds_scaled = model(X_test_tensor).detach().cpu().numpy()

    # Odskalowanie predykcji historycznych (tylko Close do wykresu)
    preds_close = preprocessor.target_scaler.inverse_transform(preds_scaled)[:, features.index("Close")]

    # Multi-step multi-feature forecast
    multi_preds = []
    last_seq = X_test[-1]
    current_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

    for step in range(n_steps):
        pred = model(current_seq).detach().cpu().numpy().flatten()  # shape (5,)
        multi_preds.append(pred)

        # sliding window - aktualizacja sekwencji
        new_step = current_seq[:, 1:, :].clone()
        next_features = torch.tensor(pred, dtype=torch.float32).unsqueeze(0).to(device)  # shape (1,5)
        current_seq = torch.cat((new_step, next_features.unsqueeze(1)), dim=1)

    multi_preds = np.array(multi_preds)
    multi_preds_true = preprocessor.target_scaler.inverse_transform(multi_preds)

    # Daty
    dates_hist = data.index[-last_days:]
    last_date = dates_hist[-1]
    dates_forecast = [last_date + timedelta(days=i + 1) for i in range(n_steps)]

    # Wykres świec
    fig, ax = plt.subplots(figsize=(14, 6))

    # Funkcja pomocnicza do rysowania świec
    def plot_candles(ax, o, h, l, c, dates, width=0.6):
        for i in range(len(dates)):
            color = 'green' if c[i] >= o[i] else 'red'
            ax.plot([dates[i], dates[i]], [l[i], h[i]], color='black')
            rect = Rectangle((mdates.date2num(dates[i]) - width/2, min(o[i], c[i])),
                             width, abs(c[i]-o[i]), color=color)
            ax.add_patch(rect)

    # Historyczne świeczki
    plot_candles(ax,
                 data['Open'].iloc[-last_days:].values,
                 data['High'].iloc[-last_days:].values,
                 data['Low'].iloc[-last_days:].values,
                 data['Close'].iloc[-last_days:].values,
                 dates_hist)

    # Prognoza historyczna LSTM (tylko Close jako linia)
    ax.plot(dates_hist[-len(preds_close):], preds_close[-len(dates_hist):], linestyle='--', color='blue',
            label='Predykcja LSTM (historyczne Close)')

    # Prognoza świecowa n dni do przodu
    plot_candles(ax,
                 multi_preds_true[:, features.index("Open")],
                 multi_preds_true[:, features.index("High")],
                 multi_preds_true[:, features.index("Low")],
                 multi_preds_true[:, features.index("Close")],
                 dates_forecast)

    ax.set_title(f"📈 Prognoza LSTM dla {ticker}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Cena [USD]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend()
    ax.grid(True)

    # Tabela z prognozami
    forecast_df = pd.DataFrame(multi_preds_true, columns=features, index=dates_forecast)
    table = plt.table(cellText=np.round(forecast_df.values, 2),
                      rowLabels=forecast_df.index.strftime('%Y-%m-%d'),
                      colLabels=forecast_df.columns,
                      cellLoc='center',
                      rowLoc='center',
                      loc='bottom',
                      bbox=[0, -0.35, 1, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.subplots_adjust(left=0.05, bottom=0.35)
    plt.show()

    return forecast_df
