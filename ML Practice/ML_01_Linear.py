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

## Scatter plot
"""plt.scatter(df['x'], df['y'], color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X vs Y')
plt.show()"""

print(df.corr())

## Independent and dependent features
X = df[['x']]
y = df['y']