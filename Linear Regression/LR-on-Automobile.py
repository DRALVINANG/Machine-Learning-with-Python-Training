#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the automobile dataset from the provided URL
url = 'https://www.alvinang.sg/s/automobileEDA.csv'
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

#--------------------------------------------------------------------
# Step 2: Define X and Y
#--------------------------------------------------------------------
# X represents the independent variable (highway-mpg), and Y represents the dependent variable (price)
X = df[['highway-mpg']]
y = df['price']

#--------------------------------------------------------------------
# Step 3: Initialize and Fit Linear Regression Model
#--------------------------------------------------------------------
# Initialize the Linear Regression model
lm = LinearRegression()

# Fit the model
lm.fit(X, y)

#--------------------------------------------------------------------
# Step 4: Visualize Price vs Highway-MPG (Regression Plot)
#--------------------------------------------------------------------
# Use Seaborn's regplot to visualize the relationship between highway-mpg and price
plt.figure(figsize=(12, 10))
sns.regplot(x='highway-mpg', y='price', data=df)
plt.title('Highway-MPG vs Price')
plt.show()

#--------------------------------------------------------------------
# Step 5: Linear Regression Equation
#--------------------------------------------------------------------
# Intercept (Y-intercept of the regression line)
print(f"Intercept: {lm.intercept_}")

# Coefficient (Slope of the regression line)
print(f"Coefficient: {lm.coef_[0]}")

#--------------------------------------------------------------------
# Step 6: Predict Prices using the Model
#--------------------------------------------------------------------
# Predict the values of price based on the model
Yhat = lm.predict(X)

# Display the first 5 predicted values
print("Predicted Prices (first 5 values):", Yhat[0:5])

#--------------------------------------------------------------------
# Step 7: Visualize Residual Plot
#--------------------------------------------------------------------
# Plotting the residuals to check for linearity
plt.figure(figsize=(12, 10))
sns.residplot(x='highway-mpg', y='price', data=df)
plt.title('Residual Plot: Highway-MPG vs Price')
plt.show()

#--------------------------------------------------------------------
# Step 8: Evaluate Model (R-Squared and MSE)
#--------------------------------------------------------------------
# R-squared value (coefficient of determination)
r2 = lm.score(X, y)
print(f"R-squared Value: {r2:.2f}")

# Mean Squared Error (MSE)
mse = mean_squared_error(y, Yhat)
print(f"Mean Squared Error: {mse:.2f}")

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------

