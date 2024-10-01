import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
advert = pd.read_csv('https://www.alvinang.sg/s/Advertising.csv')

# Preview the data
print(advert.head())

# Seaborn lmplot to show linear relationship between TV advertising and Sales
sns.lmplot(x='TV', y='Sales', data=advert, height=6, aspect=1.5)
plt.title('TV Advertising vs Sales (Seaborn lmplot)')
plt.show()

# Split data into training and testing sets
X = advert[['TV']]
y = advert['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit linear regression model using scikit-learn
model = LinearRegression()
model.fit(X_train, y_train)

# Get the model parameters (intercept and coefficient)
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Prediction for a new value (TV = $400)
new_X = 400
predicted_sales = model.predict([[new_X]])
print(f"Predicted Sales for TV = $400: {predicted_sales[0]} units")

# Predict values for all test data points
sales_pred_test = model.predict(X_test)

# Predict values for all train data points (to visualize)
sales_pred_train = model.predict(X_train)

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

# Cosmetics
plt.xlabel('TV Advertising Costs')
plt.ylabel('Sales')
plt.title('TV vs Sales (Train-Test Split)')
plt.legend()

plt.show()
