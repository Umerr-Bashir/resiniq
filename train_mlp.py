
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from split_data import X_train, y_train, X_val, y_val, X_test, y_test

model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=False,
        random_state=42
    )
)

model.fit(X_train, y_train)

val_predictions = model.predict(X_val)
predictions = model.predict(X_test)

val_mae = mean_absolute_error(y_val, val_predictions)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MLP Results")
print("Validation MAE:", round(val_mae, 3), "hours")
print("Validation RMSE:", round(val_rmse, 3), "hours")
print("Test MAE:", round(mae, 3), "hours")
print("Test RMSE:", round(rmse, 3), "hours")
