
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

# Prediction vs actual
plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.xlabel("Actual remaining time (hours)")
plt.ylabel("Predicted remaining time (hours)")
plt.title("Actual vs Predicted")
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

# Error by elapsed time
results = pd.DataFrame({
    "elapsed": X_test["elapsed"],
    "actual": y_test,
    "predicted": pred
})

results["absolute_error"] = abs(
    results["actual"] - results["predicted"]
)

results["elapsed_group"] = pd.cut(
    results["elapsed"],
    bins=[0, 2, 4, 6, 8, 100]
)

print("\nMAE by elapsed time:")
print(results.groupby(
    "elapsed_group",
    observed=True
)["absolute_error"].mean())

print("\nEvaluation complete.")
