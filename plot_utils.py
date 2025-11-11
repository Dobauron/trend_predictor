import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle

class CandlestickPlotter:
    def __init__(self, data: pd.DataFrame, features: list, preprocessor=None, model=None, X_test=None, device='cpu'):
        """
        data: pełny DataFrame z historycznymi danymi
        features: lista wszystkich cech ['Open','High','Low','Close','Volume',...]
        preprocessor: DataPreprocessor (potrzebny do odskalowania predykcji)
        model: wytrenowany model LSTM
        X_test: dane testowe (sekwencje)
        device: 'cpu' lub 'cuda'
        """
        self.data = data
        self.features = features
        self.preprocessor = preprocessor
        self.model = model
        self.X_test = X_test
        self.device = device
        self.forecast_df = None
        self.preds_close = None

    def prepare_forecast_data(self, n_steps=5):
        """Generuje prognozy multi-step i odskalowuje każdą cechę indywidualnie."""
        if self.model is None or self.X_test is None or self.preprocessor is None:
            raise ValueError("Model, X_test i preprocessor muszą być ustawione")

        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            preds_scaled = self.model(X_test_tensor).detach().cpu().numpy()

        # Odskalowanie historycznych predykcji Close
        self.preds_close = self.preprocessor.inverse_transform(
            "Close", preds_scaled[:, self.features.index("Close")]
        )

        # Multi-step forecast
        multi_preds = []
        last_seq = self.X_test[-1]
        current_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        for step in range(n_steps):
            pred = self.model(current_seq).detach().cpu().numpy().flatten()
            multi_preds.append(pred)
            new_step = current_seq[:, 1:, :].clone()
            next_features = torch.tensor(pred, dtype=torch.float32).unsqueeze(0).to(self.device)
            current_seq = torch.cat((new_step, next_features.unsqueeze(1)), dim=1)

        multi_preds = np.array(multi_preds)

        # Odskalowanie indywidualne
        multi_preds_true = np.zeros_like(multi_preds)
        print(self.features)
        for i, feat in enumerate(self.features):
            multi_preds_true[:, i] = self.preprocessor.inverse_transform(
                feature_name=feat,
                scaled_values=multi_preds[:, i]
            )

        dates_forecast = [self.data.index[-1] + timedelta(days=i + 1) for i in range(n_steps)]
        self.forecast_df = pd.DataFrame(multi_preds_true, columns=self.features, index=dates_forecast)
        return self.forecast_df, self.preds_close

    @staticmethod
    def _plot_candles(ax, o, h, l, c, dates, width=0.6):
        """Pomocnicza metoda do rysowania świec."""
        for i in range(len(dates)):
            color = 'green' if c[i] >= o[i] else 'red'
            ax.plot([dates[i], dates[i]], [l[i], h[i]], color='black')
            rect = Rectangle((mdates.date2num(dates[i]) - width/2, min(o[i], c[i])),
                             width, abs(c[i]-o[i]), color=color)
            ax.add_patch(rect)

    def plot_forecast(self, last_days=100, ticker='TICKER'):
        """Rysuje wykres świec historycznych i prognoz LSTM."""
        if self.forecast_df is None or self.preds_close is None:
            raise ValueError("Najpierw wywołaj prepare_forecast_data()")

        dates_hist = self.data.index[-last_days:]
        fig, ax = plt.subplots(figsize=(14, 6))

        # Historyczne świeczki
        self._plot_candles(ax,
                           self.data['Open'].iloc[-last_days:].values,
                           self.data['High'].iloc[-last_days:].values,
                           self.data['Low'].iloc[-last_days:].values,
                           self.data['Close'].iloc[-last_days:].values,
                           dates_hist)

        # Linia historycznej predykcji Close
        ax.plot(dates_hist[-len(self.preds_close):], self.preds_close[-len(dates_hist):],
                linestyle='--', color='blue', label='Predykcja LSTM (historyczne Close)')

        # Prognoza świecowa
        self._plot_candles(ax,
                           self.forecast_df['Open'].values,
                           self.forecast_df['High'].values,
                           self.forecast_df['Low'].values,
                           self.forecast_df['Close'].values,
                           self.forecast_df.index)

        ax.set_title(f"📈 Prognoza LSTM dla {ticker}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Cena [USD]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        ax.legend()
        ax.grid(True)

        # Tabela prognoz
        table = plt.table(cellText=np.round(self.forecast_df.values, 2),
                          rowLabels=self.forecast_df.index.strftime('%Y-%m-%d'),
                          colLabels=self.forecast_df.columns,
                          cellLoc='center',
                          rowLoc='center',
                          loc='bottom',
                          bbox=[0, -0.35, 1, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        plt.subplots_adjust(left=0.05, bottom=0.35)
        plt.show()
