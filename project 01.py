# ==========================================
# Data Science Project 1
# Advanced EDA & Feature Engineering
# ==========================================


# Import Required Libraries

# import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from scipy import stats


# ==========================================
# 1. Load Dataset
# ==========================================

# Read Excel file
df = pd.read_excel("Dataset for Data Analytics.xlsx")


# Display first 5 rows

print("First 5 Rows:")
display(df.head())


# ==========================================
# 2. Basic Exploratory Data Analysis (EDA)
# ==========================================


# Check number of rows and columns

print("Dataset Shape:")
print(df.shape)


# Display column information

print("\nDataset Information:")
df.info()


# Statistical summary

print("\nStatistical Summary:")
display(df.describe())



# ==========================================
# 3. Check Missing Values
# ==========================================


print("\nMissing Values Before Handling:")

print(df.isnull().sum())



# ==========================================
# 4. Handle Missing Values
# ==========================================


# Checking categorical columns

categorical_columns = df.select_dtypes(
    include="object"
).columns


print("\nCategorical Columns:")
print(categorical_columns)



# Fill missing categorical values using Mode

for col in categorical_columns:

    if df[col].isnull().sum() > 0:

        df[col].fillna(
            df[col].mode()[0],
            inplace=True
        )


print("\nMissing Values After Handling:")

print(df.isnull().sum())



# ==========================================
# 5. Detect Outliers Using IQR Method
# ==========================================


# Select numerical columns

numeric_columns = df.select_dtypes(
    include=np.number
).columns


print("\nNumerical Columns:")
print(numeric_columns)



# Visualize Outliers Before Removing

for col in numeric_columns:

    plt.figure(figsize=(6,3))

    sns.boxplot(
        x=df[col]
    )

    plt.title(
        "Outliers in " + col
    )

    plt.show()



# Remove Outliers Using IQR

for col in numeric_columns:


    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)


    IQR = Q3 - Q1


    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR



    df = df[
        (df[col] >= lower_limit) &
        (df[col] <= upper_limit)
    ]



print("\nShape After Removing Outliers:")

print(df.shape)



# ==========================================
# 6. Feature Engineering
# ==========================================


# Feature 1:
# Creating discount indicator feature

df["DiscountApplied"] = np.where(
    df["CouponCode"].notnull(),
    1,
    0
)



# Feature 2:
# Creating price category feature

df["PriceCategory"] = pd.cut(
    df["UnitPrice"],
    bins=[
        0,
        200,
        500,
        df["UnitPrice"].max()
    ],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)



# Feature 3:
# Calculate average price per item

df["PricePerItem"] = (
    df["TotalPrice"] /
    df["Quantity"]
)



# Feature 4:
# Extract month from Date column

if "Date" in df.columns:

    df["OrderMonth"] = (
        pd.to_datetime(df["Date"])
        .dt.month
    )



# ==========================================
# 7. Check Final Dataset
# ==========================================


print("\nFinal Dataset:")

display(df.head())



print("\nFinal Information:")

df.info()



# ==========================================
# 8. Correlation Analysis
# ==========================================


plt.figure(figsize=(10,6))


sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True
)


plt.title(
    "Correlation Heatmap"
)


plt.show()



# ==========================================
# 9. Save Clean Dataset
# ==========================================


df.to_excel(
    "Cleaned_Data_Analytics.xlsx",
    index=False
)


print(
    "Clean Dataset Saved Successfully!"
)