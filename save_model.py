
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from split_data import X_train, y_train

MODEL_PATH = "resin_model_v2.joblib"

model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(64, 32),
        alpha=0.01,
        max_iter=1000,
        early_stopping=False,
        random_state=42
    )
)

model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)

print("Model saved successfully.")
print("Saved to:", MODEL_PATH)
