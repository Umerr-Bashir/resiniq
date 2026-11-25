
import pandas as pd

from split_data import train_batches, val_batches, test_batches

df = pd.read_csv("data/rows.csv")

# 1. Batch statistics
batch_sizes = df.groupby("batch_id").size()

print("Number of batches:", df["batch_id"].nunique())
print("Minimum rows per batch:", batch_sizes.min())
print("Maximum rows per batch:", batch_sizes.max())
print("Average rows per batch:", batch_sizes.mean())

# 2. Check time_remaining consistency
df = df.sort_values(["batch_id", "elapsed"])

remaining_diff = df.groupby("batch_id")["time_remaining"].diff()

print("\nTime remaining increases:", (remaining_diff > 0).sum())
print("Time remaining stays same:", (remaining_diff == 0).sum())

# 3. Check final outcomes within batches
for col in ["final_visc", "off_spec"]:
    counts = df.groupby("batch_id")[col].nunique()
    print(f"\n{col} inconsistent batches:", (counts > 1).sum())

# 4. Split sanity checks
print("\nSplit batch counts:")
print("train:", len(train_batches))
print("validation:", len(val_batches))
print("test:", len(test_batches))
print("train/validation overlap:", len(train_batches & val_batches))
print("train/test overlap:", len(train_batches & test_batches))
print("validation/test overlap:", len(val_batches & test_batches))
