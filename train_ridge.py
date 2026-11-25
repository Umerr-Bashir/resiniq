
import numpy as np

# Extract trained Ridge model from pipeline
ridge = model.named_steps["ridge"]
scaler = model.named_steps["standardscaler"]

# Feature importance from absolute coefficients
importance = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": ridge.coef_,
    "importance": np.abs(ridge.coef_)
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature coefficients:")
print(importance.to_string(index=False))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from split_data import X_train, y_train, X_val, y_val, X_test, y_test

# Build pipeline
model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Ridge Regression Results")
print("------------------------")
print("Test MAE:", round(mae, 3), "hours")
print("Test RMSE:", round(rmse, 3), "hours")


# Extract trained Ridge model from pipeline
ridge = model.named_steps["ridge"]
scaler = model.named_steps["standardscaler"]

# Feature importance from absolute coefficients
importance = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": ridge.coef_,
    "importance": np.abs(ridge.coef_)
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature coefficients:")
print(importance.to_string(index=False))

