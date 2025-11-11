"""
Testy dla scalerów (scalers.py)
Sprawdza:
- czy każdy scaler w scaler_map działa poprawnie
- czy transformacja i odwrotna transformacja przywracają oryginalne wartości
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data_preprocessor import DataPreprocessor
from model_lstm import LSTMModel

def test_scaler_and_model_diagnostics(show_plots=True):
    """
    🔍 Test diagnostyczny:
    - Sprawdza czy skalery działają poprawnie (fit -> transform -> inverse_transform)
    - Sprawdza, czy model przewiduje dane w podobnym zakresie co oryginalne wartości
    - Pokazuje wykres oryginalnych vs. odskalowanych wartości
    """

    # --- 1️⃣ Dane testowe ---
    n_samples = 100
    features = ["Open", "High", "Low", "Close", "Volume"]
    df = pd.DataFrame(np.random.rand(n_samples, len(features))*100, columns=features)
    df.index = pd.date_range(start="2024-01-01", periods=n_samples, freq="D")

    # --- 2️⃣ Preprocessing ---
    preprocessor = DataPreprocessor(features=features, target_features=["Close"], seq_length=10)
    X_train, X_test, y_train, y_test = preprocessor.prepare(df)

    # --- 3️⃣ Test poprawności skalowania ---
    for feature in features:
        original = df[feature].values
        scaled = preprocessor.scalers[feature].transform(original.reshape(-1, 1)).flatten()
        restored = preprocessor.scalers[feature].inverse_transform(scaled.reshape(-1, 1)).flatten()

        mae_scaling = np.mean(np.abs(original - restored))
        print(f"[{feature}] błąd odskalowania: {mae_scaling:.6f}")
        assert mae_scaling < 1e-6, f"Scaler dla {feature} nie odwraca poprawnie wartości!"

    # --- 4️⃣ Model ---
    model = LSTMModel(input_size=len(features), output_size=len(features))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    preds_scaled = model(X_test_tensor).detach().numpy()

    # --- 5️⃣ Odskalowanie predykcji ---
    preds_close = preprocessor.inverse_transform("Close", preds_scaled[:, features.index("Close")])

    # --- 6️⃣ Porównanie zakresów ---
    orig_close = df["Close"].values[-len(preds_close):]
    diff_mean = np.mean(np.abs(orig_close - preds_close))

    print(f"\nŚrednia różnica między oryginalnymi a przewidzianymi (odskalowanymi): {diff_mean:.3f}")

    # --- 7️⃣ Wizualizacja ---
    if show_plots:
        plt.figure(figsize=(8, 4))
        plt.plot(orig_close, label="True Close", alpha=0.7)
        plt.plot(preds_close, label="Predicted (inverse scaled)", alpha=0.7)
        plt.title("Diagnostyka skalowania i predykcji")
        plt.legend()
        plt.grid(True)
        plt.show()

    assert not np.any(np.isnan(preds_close)), "Predykcje zawierają NaN!"
