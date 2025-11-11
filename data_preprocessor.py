import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, features: list, target_features: list, seq_length: int = 60):
        self.features = features
        self.target_features = target_features
        self.seq_length = seq_length

        # Mapa scalerów do każdej cechy
        self.scaler_map = {
            "Open": MinMaxScaler(),
            "High": MinMaxScaler(),
            "Low": MinMaxScaler(),
            "Close": MinMaxScaler(),
            "Volume": RobustScaler(),
            "RSI": MinMaxScaler(feature_range=(0, 1)),
            "MACD": StandardScaler(),
            "MACD_signal": StandardScaler(),
            "MACD_hist": StandardScaler(),
        }

        # Skalery dla cech wejściowych i celu
        self.scalers = {feat: self.scaler_map[feat] for feat in self.features}
        self.target_scalers = {feat: self.scaler_map[feat] for feat in self.target_features}

    def prepare(self, data: pd.DataFrame):
        data = data.dropna()

        # Skalowanie cech wejściowych
        scaled_features = np.zeros_like(data[self.features].values, dtype=np.float32)
        for i, feat in enumerate(self.features):
            scaled_features[:, i] = self.scalers[feat].fit_transform(
                data[[feat]]
            ).flatten()

        df_scaled = pd.DataFrame(scaled_features, columns=self.features, index=data.index)

        # Skalowanie target
        for feat in self.target_features:
            df_scaled[feat] = self.target_scalers[feat].fit_transform(
                data[[feat]]
            ).flatten()

        # Tworzenie sekwencji
        X, y = self._create_sequences(df_scaled)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
        return X_train, X_test, y_train, y_test

    def _create_sequences(self, df):
        X, y = [], []
        data_array = df[self.features].values
        target_array = df[self.target_features].values
        for i in range(len(df) - self.seq_length):
            X.append(data_array[i:i + self.seq_length])
            y.append(target_array[i + self.seq_length])
        return np.array(X), np.array(y)

    # Odskalowanie pojedynczej cechy
    def inverse_transform(self, feature_name: str, scaled_values: np.ndarray):
        return self.scalers[feature_name].inverse_transform(
            scaled_values.reshape(-1, 1)
        ).flatten()
