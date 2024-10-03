import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
# Load the Breast Cancer dataset from the provided GitHub link
url = "https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Logistic%20Regression/Breast%20Cancer.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

#--------------------------------------------------------------------
# Step 2: Train-Test Split
#--------------------------------------------------------------------
# Define X (features) and y (target)
X = df.drop(columns=['Target'])
y = df['Target']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#--------------------------------------------------------------------
# Step 3: Feature Scaling
#--------------------------------------------------------------------
# Apply StandardScaler to standardize the dataset (mean=0, variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit to training data and transform
X_test_scaled = scaler.transform(X_test)  # Only transform the test data (no fitting)

#--------------------------------------------------------------------
# Step 4: Logistic Regression Model
#--------------------------------------------------------------------
# Initialize and fit the logistic regression model
logistic_model = LogisticRegression(max_iter=5000)  # Increase max_iter to ensure convergence
logistic_model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = logistic_model.predict(X_test_scaled)
probabilities = logistic_model.predict_proba(X_test_scaled)

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
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Malignant', 'Predicted Benign'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Malignant', 'Actual Benign'))
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
# Example: Predict for a sample in the test set
example_index = 5  # Example index from the test set
example_data = X_test_scaled[example_index].reshape(1, -1)
prediction = logistic_model.predict(example_data)
probability = logistic_model.predict_proba(example_data)

print(f"Prediction for example {example_index}: {'Benign' if prediction == 1 else 'Malignant'}")
print(f"Probability of being benign: {probability[0][1]:.2f}")

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------
