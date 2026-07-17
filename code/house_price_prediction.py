import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Housing.csv")

# Display basic information
print("========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())