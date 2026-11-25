
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

from split_data import X_train, y_train, X_test, y_test

model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(64, 32),
        alpha=0.01,
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
)

model.fit(X_train, y_train)

baseline_pred = model.predict(X_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)

# Simulate +3°C temperature sensor bias
drifted = X_test.copy()
drifted["temp_last"] += 3
drifted["temp_mean"] += 3

drift_pred = model.predict(drifted)
drift_mae = mean_absolute_error(y_test, drift_pred)

print("Original MAE:", round(baseline_mae, 3))
print("Drifted MAE:", round(drift_mae, 3))
print("MAE increase:", round(drift_mae-baseline_mae, 3))
