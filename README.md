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
print(df)

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

<img width="653" height="784" alt="image" src="https://github.com/user-attachments/assets/02972252-72d0-43d5-81da-911d7a098ecf" />



<img width="620" height="438" alt="image" src="https://github.com/user-attachments/assets/7feef717-26de-4ee8-9b32-b21bf8461fb4" />



<img width="607" height="428" alt="image" src="https://github.com/user-attachments/assets/9cd5c97d-9c93-44b2-adfe-572f4ddc2aee" />



<img width="349" height="116" alt="image" src="https://github.com/user-attachments/assets/89efed0e-56ac-4802-bd7c-aa4208f93e1b" />



<img width="823" height="432" alt="image" src="https://github.com/user-attachments/assets/e7e0a2b6-b4f1-4d07-b1b0-151951dbc66a" />



<img width="436" height="334" alt="image" src="https://github.com/user-attachments/assets/3c0e66e8-c6cc-477e-8b4f-c38c53acb33e" />



<img width="1400" height="397" alt="image" src="https://github.com/user-attachments/assets/87501319-813a-44a5-b189-ad5635ce3e43" />



<img width="515" height="84" alt="image" src="https://github.com/user-attachments/assets/1d8ecf8a-f64e-4a5d-a96e-e802891703e1" />



<img width="826" height="850" alt="image" src="https://github.com/user-attachments/assets/863477f9-7221-4322-b782-7e4515b1ffa6" />



<img width="393" height="346" alt="image" src="https://github.com/user-attachments/assets/73d35747-b700-466f-923d-7df038e319a4" />



<img width="690" height="182" alt="image" src="https://github.com/user-attachments/assets/fbd50c15-f1bc-4b57-834e-4a6be08982b1" />



<img width="547" height="269" alt="image" src="https://github.com/user-attachments/assets/9e3a77c3-f4b4-447e-9319-0401959ab3e6" />



<img width="882" height="777" alt="image" src="https://github.com/user-attachments/assets/ac79f7c2-e039-4f79-9155-85de2eff5d18" />



<img width="663" height="533" alt="image" src="https://github.com/user-attachments/assets/3da5d2f6-3eac-4729-886c-993b0534c937" />



<img width="676" height="165" alt="image" src="https://github.com/user-attachments/assets/f4b64446-f47f-4ed1-a800-75b3d4a1d89b" />



<img width="553" height="161" alt="image" src="https://github.com/user-attachments/assets/fc6f4d38-cf6c-4d38-9b35-d365d62e4730" />




# RESULT :
The given dataset was successfully read, cleaned by handling missing and inconsistent data, outliers were identified and removed, and the cleaned dataset was saved to a new file successfully.


