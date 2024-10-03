#------------------------------------------------------------------------------------------------
# Step 1: Import Dataset
#------------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the wine dataset
wine = pd.read_csv("https://www.alvinang.sg/s/wine_sklearn_dataset.csv")

# Inspect the first few rows to understand the structure of the data
print(wine.head())

# The target column in the wine dataset is called 'target'
# Let's inspect the class distribution
print(wine['target'].value_counts())

#------------------------------------------------------------------------------------------------
# Step 2: Train Test Split
#------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# Target
y = wine["target"]

# Features (All columns except 'target')
X = wine.drop(columns=["target"])

# Split the data into a training set and a testing set.
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

#------------------------------------------------------------------------------------------------
# Step 3: Build and Train the RFC
#------------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier object.
rfc = RandomForestClassifier(n_estimators=100, criterion="entropy")

# Fit the model to the training data.
rfc.fit(train_X, train_y)

#------------------------------------------------------------------------------------------------
# Step 4: Prediction Comparison on Test Data
#------------------------------------------------------------------------------------------------
# Create a DataFrame to compare predicted and actual class labels for the test data
df = pd.DataFrame({
    "predicted_class": rfc.predict(test_X),
    "actual_class": test_y.tolist()
})

# Print the DataFrame to view predictions vs actual values
print(df)

#------------------------------------------------------------------------------------------------
# Step 5: Accuracy Score
#------------------------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score

# Calculate and print the accuracy of the model
accuracy = accuracy_score(test_y.tolist(), rfc.predict(test_X))
print(f"Accuracy: {accuracy * 100:.2f}%")

#------------------------------------------------------------------------------------------------
# Step 6: Visualizing the Trees in the Random Forest
#------------------------------------------------------------------------------------------------
import graphviz
from sklearn.tree import export_graphviz

# Extract individual decision trees from the Random Forest
estimators = rfc.estimators_

# Visualize each tree using Graphviz
for i in range(len(estimators)):
    dot_data = export_graphviz(estimators[i], out_file=None, feature_names=list(X.columns),
                               class_names=["Class 0", "Class 1", "Class 2"], filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    display(graph)  # This will display the decision trees

#------------------------------------------------------------------------------------------------
# Step 7: Prediction on Simulated Data
#------------------------------------------------------------------------------------------------
# Create a DataFrame with a simulated row of data
# Replace the values below with sample values for the wine dataset features
simulated_data = pd.DataFrame({
    "alcohol": [13.5],        # Example value for alcohol content
    "malic_acid": [2.3],      # Example value for malic acid
    "ash": [2.2],             # Example value for ash
    "alcalinity_of_ash": [19.5],  # Example value for alcalinity of ash
    "magnesium": [105],       # Example value for magnesium
    "total_phenols": [2.8],   # Example value for total phenols
    "flavanoids": [3.1],      # Example value for flavanoids
    "nonflavanoid_phenols": [0.3],  # Example value for nonflavanoid phenols
    "proanthocyanins": [1.9], # Example value for proanthocyanins
    "color_intensity": [4.5], # Example value for color intensity
    "hue": [1.1],             # Example value for hue
    "od280/od315_of_diluted_wines": [3.0],  # Example value for OD280/OD315
    "proline": [750]          # Example value for proline
})

# Use the trained RandomForestClassifier to predict the class of this simulated data
predicted_class = rfc.predict(simulated_data)

# Display the predicted class
print(f"Predicted class for the simulated data: {predicted_class[0]}")

#------------------------------------------------------------------------------------------------
# THE END
#------------------------------------------------------------------------------------------------
