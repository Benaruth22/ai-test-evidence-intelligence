import pandas as pd


# Load the dataset
df = pd.read_csv("data/raw/test_executions.csv")


# -----------------------------
# Basic information
# -----------------------------

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== FIRST 5 ROWS ===")
print(df.head())


# -----------------------------
# Target distribution
# -----------------------------

print("\n=== FAILURE CATEGORY COUNTS ===")
print(df["failure_category"].value_counts())

print("\n=== FAILURE CATEGORY PERCENTAGES ===")
print(
    df["failure_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# -----------------------------
# Missing values
# -----------------------------

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())


# -----------------------------
# Data types
# -----------------------------

print("\n=== DATA TYPES ===")
print(df.dtypes)


# -----------------------------
# Numerical statistics
# -----------------------------

print("\n=== NUMERICAL SUMMARY ===")
print(df.describe())