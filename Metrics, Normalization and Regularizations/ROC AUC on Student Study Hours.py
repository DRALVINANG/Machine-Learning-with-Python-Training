#-----------------------------------------------------------
# Step 1: Pip install Scikit Plot
#-----------------------------------------------------------
!pip install scikit-plot
!pip install scipy==1.7.3

#-----------------------------------------------------------
# Step 2: Import Libraries
#-----------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
import matplotlib.pyplot as plt  # Import for plotting

#-----------------------------------------------------------
# Step 3: Load the student dataset
#-----------------------------------------------------------
# Load the dataset directly from the GitHub link
df = pd.read_csv("https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Logistic%20Regression/student-study-hours.csv")

# Consider score >= 50 as "Pass" and score < 50 as "Fail"
df['Pass'] = df['Scores'] >= 50
df['Pass'] = df['Pass'].map({True: 1, False: 0})
df = df[['Hours', 'Pass']]  # Keep only 'Hours' and 'Pass' columns

# Define X (Hours) and y (Pass/Fail)
X = df[['Hours']]
y = df['Pass']

#-----------------------------------------------------------
# Step 4: Train-Test Split
#-----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#-----------------------------------------------------------
# Step 5: Train using Logistic Regression
#-----------------------------------------------------------
model = LogisticRegression(max_iter=200)  # Set max_iter to avoid convergence issues
model.fit(X_train, y_train)

#-----------------------------------------------------------
# Step 6: Display the Probabilities of each class
#-----------------------------------------------------------
y_probs = model.predict_proba(X_test)  # Get probability predictions

#-----------------------------------------------------------
# Step 7: Calculate ROC AUC Score
#-----------------------------------------------------------
roc_auc = roc_auc_score(y_test, y_probs[:, 1])  # Use probabilities for the positive class (1)
print(f"ROC AUC Score: {roc_auc:.2f}")

#-----------------------------------------------------------
# Step 8: Plotting the ROC curve
#-----------------------------------------------------------
# Plot ROC curve for binary classification
skplt.metrics.plot_roc(y_test, y_probs)
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

#-----------------------------------------------------------
# THE END
#-----------------------------------------------------------
