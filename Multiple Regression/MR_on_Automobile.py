#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Automobile dataset from the provided URL
path = 'https://www.alvinang.sg/s/automobileEDA.csv'
df = pd.read_csv(path)

# Display the first few rows of the dataset
print(df.head())

#--------------------------------------------------------------------
# Step 2: Define Features (X) and Target (y)
#--------------------------------------------------------------------
from sklearn.linear_model import LinearRegression

# Define the features (independent variables) and target (dependent variable)
X = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y = df['price']

# Initialize the Linear Regression model
lm = LinearRegression()

# Fit the model to the data
lm.fit(X, y)

#--------------------------------------------------------------------
# Step 3: Model Parameters (Intercept and Coefficients)
#--------------------------------------------------------------------
# Display the intercept rounded to 2 decimal places
print(f"Intercept: {round(lm.intercept_, 2)}")

# Display the coefficients rounded to 2 decimal places
print("Coefficients:")
for feature, coef in zip(X.columns, lm.coef_):
    print(f"{feature}: {round(coef, 2)}")


#--------------------------------------------------------------------
# Step 4: Make Predictions (Fitted Values)
#--------------------------------------------------------------------
# Predict the target values based on the features
y_pred = lm.predict(X)

# Display the first few predicted values with 2 decimal places, each on a new line
print("First 5 predicted prices:")
for price in y_pred[:5]:
    print(f"{price:.2f}")

#--------------------------------------------------------------------
# Step 5: Visualize Actual vs Fitted Values (Distribution Plot)
#--------------------------------------------------------------------
import seaborn as sns

# Plot the actual vs fitted values
plt.figure(figsize=(12, 8))
sns.distplot(df['price'], hist=False, color='r', label='Actual Price')
sns.distplot(y_pred, hist=False, color='b', label='Fitted Price')
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()

#--------------------------------------------------------------------
# Step 6: Residual Plot (Check for Homoscedasticity)
#--------------------------------------------------------------------
# Plot the residuals (difference between actual and predicted)
plt.figure(figsize=(12, 8))
sns.residplot(x=y_pred, y=y, lowess=True, color='g')
plt.xlabel('Predicted Price')  # Updated label for better understanding
plt.ylabel('Residuals')
plt.title('Residual Plot: Predicted Price vs Residuals')
plt.show()


#--------------------------------------------------------------------
# Step 7: Evaluate the Model (R-squared and MSE)
#--------------------------------------------------------------------
from sklearn.metrics import mean_squared_error

# Calculate the R-squared value
r2 = lm.score(X, y)
print(f"R-squared Value: {r2:.2f}")

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------

