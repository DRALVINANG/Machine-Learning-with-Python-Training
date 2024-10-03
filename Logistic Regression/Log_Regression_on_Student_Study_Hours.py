import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
# Load the dataset directly from the GitHub link
df = pd.read_csv("https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Logistic%20Regression/student-study-hours.csv")

# Check the dataset
print(df.head())

# Consider score >= 50 as "Pass" and score < 50 as "Fail"
df['Pass'] = df['Scores'] >= 50

# Mapping Pass/Fail to 1 and 0
df['Pass'] = df['Pass'].map({True: 1, False: 0})

# Drop any unnecessary columns (if present, StudentId)
df = df[['Hours', 'Pass']]

print(df)

#--------------------------------------------------------------------
# Step 2: Train-Test Split
#--------------------------------------------------------------------
# Define X (Hours) and y (Pass/Fail)
X = df[['Hours']]
y = df['Pass']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#--------------------------------------------------------------------
# Step 3: Logistic Regression Model
#--------------------------------------------------------------------
# Initialize and fit the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = logistic_model.predict(X_test)
probabilities = logistic_model.predict_proba(X_test)

#--------------------------------------------------------------------
# Step 4: Visualize Results (Logistic Curve and Decision Boundary)
#--------------------------------------------------------------------
# Generate a smooth range of values for study hours for prediction
X_new = np.linspace(df['Hours'].min(), df['Hours'].max(), 300).reshape(-1, 1)

# Predict probabilities for this range of hours
y_logistic_pred_smooth = logistic_model.predict_proba(X_new)[:, 1]

# Find the decision boundary where probability is 0.5
coef = logistic_model.coef_[0]  # coefficient m
intercept = logistic_model.intercept_[0]  # intercept c
decision_boundary = -intercept / coef

# Plotting the logistic regression curve
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train['Hours'], y=y_train, color='blue', label='Train Data (Pass/Fail)')
sns.lineplot(x=X_new.flatten(), y=y_logistic_pred_smooth, color='red', label='Logistic Regression Curve')

# Add vertical and horizontal lines for probability = 0.5 and decision boundary
plt.axhline(y=0.5, color='green', linestyle='--', label='Probability = 0.5')
plt.axvline(x=decision_boundary[0], color='purple', linestyle='--', label=f'Decision Boundary = {decision_boundary[0]:.2f} hours')

# Adding labels and title
plt.title('Logistic Regression: Hours vs Pass/Fail')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.legend()
plt.grid(True)
plt.show()

print(f"The number of hours where the model predicts a 'pass': {decision_boundary[0]:.2f} hours")

#--------------------------------------------------------------------
# Step 5: Model Evaluation
#--------------------------------------------------------------------
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plotting confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Fail', 'Predicted Pass'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Fail', 'Actual Pass'))
ax.set_ylim(1.5, -0.5)

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.show()

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

#--------------------------------------------------------------------
# Step 6: Prediction Examples
#--------------------------------------------------------------------
# Example: Predict pass/fail for a student who studied 7 hours
hours_studied = np.array([[7]])
prediction = logistic_model.predict(hours_studied)
probability = logistic_model.predict_proba(hours_studied)

print(f"Prediction for 7 hours of study: {'Pass' if prediction == 1 else 'Fail'}")
print(f"Probability of passing: {probability[0][1]:.2f}")

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------
