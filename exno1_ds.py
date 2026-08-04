import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read Dataset
df = pd.read_csv("Loan_data.csv")

print("Dataset Loaded Successfully!")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
df.info()

print("\nDataset Shape")
print(df.shape)

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------
# Data Cleaning
# -----------------------------

# Fill missing values
df = df.fillna(method='ffill')

# Remove remaining missing values (if any)
df = df.dropna()

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# Remove Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("\nDataset After Removing Duplicates")
print(df)

print("\nData Types")
print(df.dtypes)

print("\nColumn Names")
print(df.columns)

# -----------------------------
# Modified Z-Score Method
# -----------------------------

column = "LoanAmount"

plt.figure(figsize=(6,4))
plt.boxplot(df[column])
plt.title("Before Outlier Removal")
plt.ylabel(column)
plt.show()

# Median
median = df[column].median()

# Median Absolute Deviation (MAD)
MAD = np.median(np.abs(df[column] - median))

print("Median =", median)
print("MAD =", MAD)

# Modified Z-Score
modified_z = 0.6745 * (df[column] - median) / MAD

# Detect Outliers
outliers = df[np.abs(modified_z) > 3.5]

print("\nDetected Outliers")
print(outliers)

print("\nNumber of Outliers:", len(outliers))

# Remove Outliers
df_cleaned = df[np.abs(modified_z) <= 3.5]

plt.figure(figsize=(6,4))
plt.boxplot(df_cleaned[column])
plt.title("After Outlier Removal")
plt.ylabel(column)
plt.show()

print("\nOriginal Dataset Size:", df.shape)
print("Cleaned Dataset Size:", df_cleaned.shape)
print("Number of Outliers Removed:", len(df) - len(df_cleaned))

# Save Cleaned Dataset
df_cleaned.to_csv("cleaned_loan_data.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")
