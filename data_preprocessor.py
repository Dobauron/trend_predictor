import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, features: list, seq_length: int = 30):
        self.features = features
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()

    def prepare(self, data: pd.DataFrame):
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        data.dropna(inplace=True)

        scaled_features = self.scaler.fit_transform(data[self.features])
        df_scaled = pd.DataFrame(scaled_features, columns=self.features, index=data.index)
        df_scaled["Target"] = data["Target"]

        X, y = self._create_sequences(df_scaled)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test

    def _create_sequences(self, df):
        X, y = [], []
        data_array = df[self.features].values
        target_array = df["Target"].values
        for i in range(len(df) - self.seq_length):
            X.append(data_array[i:i + self.seq_length])
            y.append(target_array[i + self.seq_length])
        return np.array(X), np.array(y)
