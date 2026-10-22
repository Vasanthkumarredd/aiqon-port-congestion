import pandas as pd
import numpy as np

df = pd.read_csv("/Users/vasanth/Downloads/guam_2025.csv")


print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types and missing values:")
df.info()

print("\nFirst five rows:")
print(df.head())

print("\nStatistical summary:")
print(df.describe(include="all").T)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())