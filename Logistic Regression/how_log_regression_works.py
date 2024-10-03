import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
# Load the dataset directly from the GitHub link
df = pd.read_csv("https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Logistic%20Regression/student-study-hours.csv")

# Check the dataset
print(df.head())

#--------------------------------------------------------------------
# Step 2: Linear Regression
#--------------------------------------------------------------------
# Perform Linear Regression
X = df[['Hours']]
y = df['Scores']

# Fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Make predictions using the linear regression model
y_pred = linear_model.predict(X)

# Find the number of hours required to achieve a score of 50
required_hours = (50 - linear_model.intercept_) / linear_model.coef_[0]

# Plot the linear regression
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Scores')
plt.plot(X, y_pred, color='red', label=f'Linear Regression Line\ny = {linear_model.coef_[0]:.2f}x + {linear_model.intercept_:.2f}')

# Add vertical and horizontal lines for score = 50
plt.axhline(y=50, color='green', linestyle='--', label='Score = 50 (Passing Mark)')
plt.axvline(x=required_hours, color='purple', linestyle='--', label=f'Hours = {required_hours:.2f}')

# Adding labels and title
plt.title('Linear Regression: Hours vs Scores with Passing Mark')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.legend()
plt.grid(True)
plt.show()

print(f"To score 50, the required number of study hours is approximately: {required_hours:.2f} hours")


#--------------------------------------------------------------------
# Step 3: Logistic Regression
#--------------------------------------------------------------------
# Now, we will perform logistic regression to predict pass/fail
# Consider score >= 50 as "Pass" and score < 50 as "Fail"
df['Pass'] = df['Scores'] >= 50

# Encode the Pass/Fail labels
le = LabelEncoder()
y_logistic = le.fit_transform(df['Pass'])

# Perform Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X, y_logistic)

# Generate a smooth range of values for study hours
X_new = np.linspace(df['Hours'].min(), df['Hours'].max(), 300).reshape(-1, 1)

# Predict probabilities using the logistic regression model for this range of values
y_logistic_pred_smooth = logistic_model.predict_proba(X_new)[:, 1]

#--------------------------------------------------------------------
# Step 4: Find Decision Boundary and Plot
#--------------------------------------------------------------------
# Find the decision boundary for logistic regression
coef = logistic_model.coef_[0]  # coefficient m
intercept = logistic_model.intercept_[0]  # intercept c

# Solving for x when the decision boundary is at probability 0.5 (logistic regression output = 0)
decision_boundary = -intercept / coef

# Plotting with Seaborn for Logistic Regression
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Hours'], y=y_logistic, color='blue', label='Pass/Fail (1 = Pass, 0 = Fail)')
sns.lineplot(x=X_new.flatten(), y=y_logistic_pred_smooth, color='red', label='Logistic Regression Curve')

# Add vertical and horizontal lines to mark the decision boundary at probability 0.5
plt.axhline(y=0.5, color='green', linestyle='--', label='Probability = 0.5')
plt.axvline(x=decision_boundary[0], color='purple', linestyle='--', label=f'Decision Boundary = {decision_boundary[0]:.2f} hours')

# Adding labels and title
plt.title('Logistic Regression: Hours vs Pass/Fail (Smoothed)')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.legend()
plt.grid(True)
plt.show()

print(f"The number of hours where the model moves from 'fail' to 'pass': {decision_boundary[0]:.2f} hours")

