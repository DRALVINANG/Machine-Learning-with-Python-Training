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
# Step 3: Logistic Regression Model
#--------------------------------------------------------------------
# Initialize and fit the logistic regression model
logistic_model = LogisticRegression(max_iter=5000)  # Increase max_iter to ensure convergence
logistic_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = logistic_model.predict(X_test)
probabilities = logistic_model.predict_proba(X_test)

#--------------------------------------------------------------------
# Step 4: Visualize Results (Logistic Curve and Decision Boundary)
#--------------------------------------------------------------------
# We will skip this step for breast cancer since it's a high-dimensional dataset (30 features), and plotting 
# a logistic curve would not be practical in this case. Logistic regression is trained on all features.

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
example_data = X_test.iloc[example_index].values.reshape(1, -1)
prediction = logistic_model.predict(example_data)
probability = logistic_model.predict_proba(example_data)

print(f"Prediction for example {example_index}: {'Benign' if prediction == 1 else 'Malignant'}")
print(f"Probability of being benign: {probability[0][1]:.2f}")

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------

