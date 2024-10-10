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
# Visualize pairplots to observe relationships between all features and sales
sns.pairplot(advert)
plt.show()

#--------------------------------------------------------------------
# Step 3: Train-Test Split
#--------------------------------------------------------------------
# Define X (features) and y (target)
X = advert[['TV', 'Radio', 'Newspaper']]
y = advert['Sales']

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#--------------------------------------------------------------------
# Step 4: Multiple Linear Regression Model
#--------------------------------------------------------------------
# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the model parameters (intercept and coefficients)
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {list(zip(X.columns, model.coef_))}")

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
# Example: Predict sales for a given set of advertising values (TV = 400, Radio = 50, Newspaper = 20)
new_X = [[400, 50, 20]]
predicted_sales = model.predict(new_X)
print(f"Predicted Sales for TV = $400, Radio = $50, Newspaper = $20: {predicted_sales[0]} units")

#--------------------------------------------------------------------
# Step 7: Visualize Results (Regression Results)
#--------------------------------------------------------------------
# Visualize residuals to understand how well the model fits
plt.figure(figsize=(12, 6))

# Plot residuals
sns.residplot(x=y_test, y=sales_pred_test, lowess=True, color="g")

# Add labels, title, and legend for clarity
plt.xlabel('Actual Sales')
plt.ylabel('Residuals')
plt.title('Residual Plot: Actual Sales vs Predicted Sales')

# Display the plot
plt.show()

#--------------------------------------------------------------------
# Step 8: R-Squared Value and Other Metrics
#--------------------------------------------------------------------
# Calculate the R-squared value to evaluate the model
r2 = r2_score(y_test, sales_pred_test)
print(f"R-squared Value: {r2:.2f}")

#--------------------------------------------------------------------
# Step 9: 3D Plot for TV, Radio, and Sales (Optional)
#--------------------------------------------------------------------
# Plotting 3D graph for two features (TV, Radio) vs Sales
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(advert['TV'], advert['Radio'], advert['Sales'], color='b')

# Labels and title
ax.set_xlabel('TV Advertising Costs ($)')
ax.set_ylabel('Radio Advertising Costs ($)')
ax.set_zlabel('Sales (units)')
plt.title('3D Plot of TV, Radio Advertising vs Sales')

plt.show()

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------
