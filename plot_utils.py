import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle

class CandlestickPlotter:
    """
    Klasa do przygotowania prognoz i rysowania wykresów świecowych.
    Obsługuje scenariusz, w którym model LSTM widzi wskaźniki dodatkowe,
    ale przewiduje tylko wybrane target_features.
    """

    def __init__(self, data: pd.DataFrame, features: list, preprocessor, model, X_test, device=None):
        self.data = data
        self.features = features
        self.preprocessor = preprocessor
        self.model = model
        self.X_test = X_test
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preds_close = None

    def prepare_forecast_data(self, n_steps=5, target_features=None):
        """
        Generuje predykcje multi-step oraz odskalowuje je.
        target_features - lista cech, które model faktycznie przewiduje (np. Open, High, Low, Close, Volume)
        """
        target_features = target_features or ["Open", "High", "Low", "Close", "Volume"]

        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            preds_scaled = self.model(X_test_tensor).detach().cpu().numpy()

        # Odskalowanie historycznych predykcji dla target_features
        close_idx = self.features.index("Close")
        self.preds_close = self.preprocessor.inverse_transform(
            "Close", preds_scaled[:, close_idx]
        )

        # Multi-step forecast
        multi_preds = []
        last_seq = self.X_test[-1]  # (seq_len, n_features)
        current_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        for step in range(n_steps):
            pred = self.model(current_seq).detach().cpu().numpy().flatten()  # shape (len(target_features),)
            multi_preds.append(pred)

            # sliding window - aktualizacja sekwencji
            new_step = current_seq[:, 1:, :].clone()  # usuń pierwszy dzień
            next_day = new_step[:, -1, :].clone()     # weź ostatni dzień sekwencji (wszystkie cechy)
            for i, feat in enumerate(target_features):
                idx = self.features.index(feat)
                next_day[:, idx] = torch.tensor(pred[i], dtype=torch.float32).to(self.device)
            current_seq = torch.cat((new_step, next_day.unsqueeze(1)), dim=1)

        multi_preds = np.array(multi_preds)

        # Odskalowanie predykcji do rzeczywistych wartości
        multi_preds_true = np.zeros_like(multi_preds)
        for i, feat in enumerate(target_features):
            multi_preds_true[:, i] = self.preprocessor.inverse_transform(feat, multi_preds[:, i])

        # Daty dla prognozy
        last_date = self.data.index[-1]
        dates_forecast = [last_date + timedelta(days=i + 1) for i in range(n_steps)]

        forecast_df = pd.DataFrame(multi_preds_true, columns=target_features, index=dates_forecast)
        return forecast_df, self.preds_close

    def plot_forecast(self, forecast_df, preds_close, last_days=100, ticker="TICKER"):
        """
        Rysuje wykres świec historycznych oraz prognozowanych.
        """
        # Daty historyczne
        dates_hist = self.data.index[-last_days:]

        fig, ax = plt.subplots(figsize=(14, 6))

        # Funkcja do rysowania świec
        def plot_candles(ax, o, h, l, c, dates, width=0.6):
            for i in range(len(dates)):
                color = 'green' if c[i] >= o[i] else 'red'
                ax.plot([dates[i], dates[i]], [l[i], h[i]], color='black')
                rect = Rectangle((mdates.date2num(dates[i]) - width/2, min(o[i], c[i])),
                                 width, abs(c[i]-o[i]), color=color)
                ax.add_patch(rect)

        # Historyczne świeczki
        plot_candles(ax,
                     self.data['Open'].iloc[-last_days:].values,
                     self.data['High'].iloc[-last_days:].values,
                     self.data['Low'].iloc[-last_days:].values,
                     self.data['Close'].iloc[-last_days:].values,
                     dates_hist)

        # Historyczne predykcje LSTM dla Close
        ax.plot(dates_hist[-len(preds_close):], preds_close[-len(dates_hist):],
                linestyle='--', color='blue', label='Predykcja LSTM (Close)')

        # Prognoza świecowa n dni do przodu
        plot_candles(ax,
                     forecast_df['Open'].values,
                     forecast_df['High'].values,
                     forecast_df['Low'].values,
                     forecast_df['Close'].values,
                     forecast_df.index)

        ax.set_title(f"📈 Prognoza LSTM dla {ticker}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Cena [USD]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        ax.legend()
        ax.grid(True)

        # Tabela z prognozami
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
