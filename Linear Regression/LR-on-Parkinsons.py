#------------------------------------------------------------------------------------------------
# Step 1: Import Dataset
#------------------------------------------------------------------------------------------------
import pandas as pd

# Load the Parkinson's dataset
dataset_path = "https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Linear%20Regression/Parkinsons.csv"
data = pd.read_csv(dataset_path)
data.head()

#------------------------------------------------------------------------------------------------
# Step 2: Extract Features and Target
#------------------------------------------------------------------------------------------------
# Drop any rows with missing values if present
data = data.dropna()

# Separate features (X) and target (y)
X = data.drop(columns=['total_UPDRS'])
y = data['total_UPDRS']

#------------------------------------------------------------------------------------------------
# Step 3: Scale Features
#------------------------------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

# Scale features to a range between 0 and 1
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#------------------------------------------------------------------------------------------------
# Step 4: Split Data into Training and Testing Sets
#------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# Split data with 70% for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=100)

#------------------------------------------------------------------------------------------------
# Step 5: Create and Train Linear Regression Model
#------------------------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

#------------------------------------------------------------------------------------------------
# Step 6: Predict and Evaluate Model Performance
#------------------------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print("Mean Squared Error (Testing):", mse)
print("R-squared (Testing):", r_squared)

#------------------------------------------------------------------------------------------------
# Step 7: Visualize Predictions vs Actual Values
#------------------------------------------------------------------------------------------------
# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual UPDRS")
plt.ylabel("Predicted UPDRS")
plt.title("Predicted vs Actual UPDRS")
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([0, 100], [0, 100], 'r')
plt.show()
#------------------------------------------------------------------------------------------------
# THE END
#------------------------------------------------------------------------------------------------
