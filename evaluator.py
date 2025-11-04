import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ModelEvaluator:
    @staticmethod
    def evaluate(y_test, y_pred):
        print("\n✅ Wyniki modelu:")
        print("Dokładność:", round(accuracy_score(y_test, y_pred), 3))
        print("\nRaport klasyfikacji:\n", classification_report(y_test, y_pred))
        print("Macierz pomyłek:\n", confusion_matrix(y_test, y_pred))

    @staticmethod
    def plot_training(history):
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['accuracy'], label='Train acc')
        plt.plot(history.history['val_accuracy'], label='Val acc')
        plt.title("Dokładność treningu")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_trends(data, y_test, y_pred, ticker):
        plt.figure(figsize=(10, 5))
        plt.plot(data.index[-len(y_test):], y_test, label='Rzeczywisty trend', alpha=0.7)
        plt.plot(data.index[-len(y_test):], y_pred, label='Prognozowany trend', alpha=0.7)
        plt.title(f"📊 Trend rzeczywisty vs prognozowany – {ticker}")
        plt.legend()
        plt.show()
