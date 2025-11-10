import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle
from datetime import timedelta

def plot_candlestick_forecast(data, preds_true, preds_close, features, target_features,
                              last_days=100, n_steps=5, ticker='TICKER'):
    """
    Wizualizuje dane historyczne (świece), predykcję historyczną (linia) i prognozę przyszłych dni (świece),
    a pod wykresem wyświetla tabelę z wartościami prognozy.
    """
    # Daty
    dates_hist = data.index[-last_days:]
    last_date = dates_hist[-1]
    dates_forecast = [last_date + timedelta(days=i + 1) for i in range(n_steps)]

    # --- Ustawienie figure: 2 wiersze, 1 kolumna ---
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])  # wykres
    ax2 = fig.add_subplot(gs[1])  # tabela

    # --- Funkcja pomocnicza do rysowania świec ---
    def plot_candles(ax, o, h, l, c, dates, width=0.6):
        for i in range(len(dates)):
            color = 'green' if c[i] >= o[i] else 'red'
            ax.plot([dates[i], dates[i]], [l[i], h[i]], color='black')
            rect = Rectangle(
                (mdates.date2num(dates[i]) - width / 2, min(o[i], c[i])),
                width,
                abs(c[i] - o[i]),
                color=color
            )
            ax.add_patch(rect)

    # --- Historyczne świeczki ---
    plot_candles(
        ax1,
        data['Open'].iloc[-last_days:].values,
        data['High'].iloc[-last_days:].values,
        data['Low'].iloc[-last_days:].values,
        data['Close'].iloc[-last_days:].values,
        dates_hist
    )

    # --- Predykcja LSTM (tylko linia Close) ---
    ax1.plot(
        dates_hist[-len(preds_close):],
        preds_close[-len(dates_hist):],
        linestyle='--',
        color='blue',
        label='Predykcja LSTM (historyczne Close)'
    )

    # --- Prognoza świecowa ---
    plot_candles(
        ax1,
        preds_true[:, features.index("Open")],
        preds_true[:, features.index("High")],
        preds_true[:, features.index("Low")],
        preds_true[:, features.index("Close")],
        dates_forecast
    )

    # --- Ustawienia wykresu ---
    ax1.set_title(f"📈 Prognoza LSTM dla {ticker}")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Cena [USD]")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.legend()
    ax1.grid(True)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # --- Tabela pod wykresem ---
    forecast_df = pd.DataFrame(preds_true[:, :len(target_features)],
                               columns=target_features,
                               index=dates_forecast)
    forecast_df_rounded = np.round(forecast_df.values, 2)

    ax2.axis("off")
    table = ax2.table(
        cellText=forecast_df_rounded,
        rowLabels=forecast_df.index.strftime('%Y-%m-%d'),
        colLabels=forecast_df.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.tight_layout()
    plt.show()

    return forecast_df
