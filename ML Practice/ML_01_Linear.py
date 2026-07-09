## Simple Linear Regression Example 
print("Program started")
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('linear_regression_dataset.csv')
print(df.head())

## Check for null values
print(df.isnull().sum())
print(df.describe())