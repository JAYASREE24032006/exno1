# EX - 1 DATA CLEANING PROCESS USING PYTHON

## AIM :
To load the given dataset, perform data cleaning by handling missing and inconsistent data, detect and remove outliers, and save the cleaned dataset as a new file.

## EXPLANATION :

### Data cleaning :
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

### Outlier Finding :
Outlier finding is the process of identifying data points that differ significantly from the rest of the dataset. These unusual values may occur due to measurement errors, data entry mistakes, or natural variations and can negatively affect the accuracy of data analysis and machine learning models.

### Outlier Removal :
Outlier removal is the process of eliminating or treating these extreme values to improve the quality, consistency, and reliability of the dataset. By removing or handling outliers appropriately, the dataset becomes more representative of the underlying data, leading to more accurate analysis and better model performance.

# ALGORITHM :

STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# PROGRAM :

```python
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

```



# OUTPUT:


# RESULT :
The given dataset was successfully read, cleaned by handling missing and inconsistent data, outliers were identified and removed, and the cleaned dataset was saved to a new file successfully.


