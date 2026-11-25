import pandas as pd

df = pd.read_csv("data/rows.csv")

# print("Dataset shape:", df.shape)
#
# print("\nColumns:")
# print(df.columns.tolist())
#
# print("\nData types:")
# print(df.dtypes)
#
# print("\nFirst 5 rows:")
# print(df.head())
#
# print("\nMissing values:")
# print(df.isnull().sum())
#
# print("\nBasic statistics:")
# print(df.describe())

print("Unique batches:", df["batch_id"].nunique())

# Feature and Predict
FEATURES = [
    "elapsed",
    "av_init",
    "temp_set",
    "catalyst",
    "glycerol_ratio",
    "purity_meas",
    "lab_av",
    "lab_av_prev",
    "lab_visc",
    "torque",
    "torque_slope",
    "temp_last",
    "temp_mean",
    "water_cum",
    "water_rate"
]

TARGET = "time_remaining"

