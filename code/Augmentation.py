import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("Dell_systems.csv")

# Columns to augment
columns = [
    "manufacturer", "category", "subcategory",
    "gwp_total", "gwp_use_ratio", "lifetime",
    "gwp_manufacturing_ratio", "screen_size", "weight"
]

# Drop missing values and select only necessary columns
df_clean = df[columns].dropna().reset_index(drop=True)

# Define how many synthetic samples you want (e.g., 3x of original)
num_samples = len(df_clean) * 3

# Separate categorical and numerical columns
categorical_cols = ["manufacturer", "category", "subcategory"]
numerical_cols = [
    "gwp_total", "gwp_use_ratio", "lifetime",
    "gwp_manufacturing_ratio", "screen_size", "weight"
]

# Sample random rows for base generation
sampled = df_clean.sample(n=num_samples, replace=True).reset_index(drop=True)

# Add small Gaussian noise to numeric columns
for col in numerical_cols:
    std_dev = df_clean[col].std()
    noise = np.random.normal(loc=0, scale=0.02 * std_dev, size=num_samples)
    sampled[col] = sampled[col] + noise
    sampled[col] = sampled[col].clip(lower=0) 

sampled["screen_size"] = sampled["screen_size"].round(1)
sampled["weight"] = sampled["weight"].round(2)

sampled.to_csv("Dell_systems.csv", index=False)

print(sampled.head())
