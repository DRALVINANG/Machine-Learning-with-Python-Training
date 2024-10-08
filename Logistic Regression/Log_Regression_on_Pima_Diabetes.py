import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
# Load the Diabetes Pima dataset
url = "https://www.alvinang.sg/s/diabetespima.csv"  # Assuming this is the correct file
df = pd.read_csv(url)

# Rename columns to reflect the actual dataset headers
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
              'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Display the first few rows of the dataset
print(df.head())

#--------------------------------------------------------------------
# Step 2: Exploratory Data Analysis with regplots
#--------------------------------------------------------------------
# Plotting regplots for all key features against the 'Outcome' (diabetes) outcome
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age']

for feature in features:
    sns.regplot(x=feature, y='Outcome', data=df, logistic=True, ci=None, y_jitter=0.03)
    plt.title(f'Regplot: {feature} vs Outcome (Diabetes)')
    plt.xlabel(f'{feature}')
    plt.ylabel('Outcome (Diabetes)')
    plt.show()

#--------------------------------------------------------------------
# Step 3: Select Features and Outcome
#--------------------------------------------------------------------
# We use only 'Glucose' and 'BMI' for this specific analysis
X = df[['Glucose', 'BMI']]
y = df['Outcome']

#--------------------------------------------------------------------
# Step 4: Train-Test Split
#--------------------------------------------------------------------
# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#--------------------------------------------------------------------
# Step 5: Feature Scaling
#--------------------------------------------------------------------
# Apply StandardScaler to standardize the dataset (mean=0, variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit to training data and transform
X_test_scaled = scaler.transform(X_test)  # Only transform the test data (no fitting)

#--------------------------------------------------------------------
# Step 6: Logistic Regression Model
#--------------------------------------------------------------------
# Initialize and fit the logistic regression model
logistic_model = LogisticRegression(max_iter=5000)  # Increase max_iter to ensure convergence
logistic_model.fit(X_train_scaled, y_train)

#--------------------------------------------------------------------
# Step 7: Visualizing the Decision Boundary
#--------------------------------------------------------------------
# Create a meshgrid to plot the decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict over the grid
Z = logistic_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('purple', 'green')))

# Plot the actual data points
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolor='k', 
            cmap=ListedColormap(('purple', 'green')))
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Glucose (Standardized)')
plt.ylabel('BMI (Standardized)')
plt.colorbar()
plt.show()

#--------------------------------------------------------------------
# Step 8: Model Evaluation
#--------------------------------------------------------------------
# Predictions on the test set
y_pred = logistic_model.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------
