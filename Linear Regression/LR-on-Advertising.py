#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  # Import r2_score for R-squared

# Load the Advertising dataset from the provided URL
url = 'https://www.alvinang.sg/s/Advertising.csv'
advert = pd.read_csv(url)

# Display the first few rows of the dataset
print(advert.head())

#--------------------------------------------------------------------
# Step 2: Exploratory Data Analysis (EDA)
#--------------------------------------------------------------------
# Visualize the linear relationship between TV advertising and Sales using Seaborn lmplot
sns.lmplot(x='TV', y='Sales', data=advert, height=6, aspect=1.5)
plt.title('TV Advertising vs Sales (Seaborn lmplot)')
plt.show()

#--------------------------------------------------------------------
# Step 3: Train-Test Split
#--------------------------------------------------------------------
# Define X (features) and y (target)
X = advert[['TV']]
y = advert['Sales']

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#--------------------------------------------------------------------
# Step 4: Linear Regression Model
#--------------------------------------------------------------------
# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the model parameters (intercept and coefficient)
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

#--------------------------------------------------------------------
# Step 5: Predictions on Test Set
#--------------------------------------------------------------------
# Predict values for the test set
sales_pred_test = model.predict(X_test)

# Predict values for the training set (to visualize later)
sales_pred_train = model.predict(X_train)

#--------------------------------------------------------------------
# Step 6: Model Evaluation (Example Prediction)
#--------------------------------------------------------------------
# Example: Predict sales for TV advertising cost of $400
new_X = [[400]]
predicted_sales = model.predict(new_X)
print(f"Predicted Sales for TV = $400: {predicted_sales[0]} units")

#--------------------------------------------------------------------
# Step 7: Visualize Results (Regression Line)
#--------------------------------------------------------------------
# Plot regression line against actual data for both train and test sets
plt.figure(figsize=(12, 6))

# Scatter plot showing actual training data
plt.scatter(X_train, y_train, label='Training Data', color='blue')

# Regression line for training data
plt.plot(X_train, sales_pred_train, 'r', label='Regression Line (Train)', linewidth=2)

# Scatter plot showing actual testing data
plt.scatter(X_test, y_test, label='Testing Data', color='green')

# Regression line for test data (though it's the same since it's the same model)
plt.plot(X_test, sales_pred_test, 'orange', label='Regression Line (Test)', linewidth=2)

# Add labels, title, and legend for clarity
plt.xlabel('TV Advertising Costs ($)')
plt.ylabel('Sales (units)')
plt.title('TV Advertising vs Sales (Train-Test Split)')
plt.legend()

# Display the plot
plt.show()

#--------------------------------------------------------------------
# Step 8: R-Squared Value
#--------------------------------------------------------------------
# Calculate the R-squared value to evaluate the model
r2 = r2_score(y_test, sales_pred_test)
print(f"R-squared Value: {r2:.2f}")

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------
