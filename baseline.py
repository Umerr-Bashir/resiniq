
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv("data/rows.csv")

# Same split as our previous script
from split_data import X_train, y_train, X_val, y_val, X_test, y_test

# Calculate average remaining time using training data only
average_time = y_train.mean()

# Predict the same value for every test observation
predictions = np.full(len(y_test), average_time)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Baseline prediction:", round(average_time, 3), "hours")
print("Test MAE:", round(mae, 3), "hours")
print("Test RMSE:", round(rmse, 3), "hours")
