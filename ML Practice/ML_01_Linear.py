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
df = df.dropna()
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

## Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
## Train Model 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

## Pridictions 
pred = reg.predict(X_test)
print(pred[:5])

"""from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, pred)
print("MSE :",mse)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, pred)
print("R2 score", r2)"""
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred
})

print(comparison)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, pred)
print("MSE :",mse)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, pred)
print("R2 score", r2)
"""plt.scatter(y_test, pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()"""
## Plot the Training data best fit line
plt.scatter(X_train,y_train)
plt.plot(X_train,reg.predict(X_train))
plt.show()