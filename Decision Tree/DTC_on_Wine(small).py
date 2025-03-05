#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from graphviz import Source

# Load the wine dataset
wine = pd.read_csv("https://www.alvinang.sg/s/wine_small.csv")

# Display the first few rows of the dataset to verify loading
print(wine.head(), '\n\n')

#--------------------------------------------------------------------
# Step 2: Train Test Split
#--------------------------------------------------------------------
# Target (the wine class)
y = wine["target"]

# Features (alcohol, flavanoids, color_intensity)
X = wine[["alcohol", "flavanoids", "color_intensity"]]

# Split the data into a training set and a testing set (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#--------------------------------------------------------------------
# Step 3: Train the Decision Tree
#--------------------------------------------------------------------
# Build the Decision Tree Classifier
dtc = tree.DecisionTreeClassifier(criterion='entropy')

# Train the classifier
dtc.fit(X_train, y_train)

#--------------------------------------------------------------------
# Step 4: Testing
#--------------------------------------------------------------------
# Predict on the test set and create a DataFrame to compare predictions vs actual classes
df = pd.DataFrame({
    "predicted_class": dtc.predict(X_test),
    "actual_class": y_test.tolist()
})

# Print the comparison DataFrame
print(df, '\n\n')

#--------------------------------------------------------------------
# Step 5: Accuracy Score
#--------------------------------------------------------------------
# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, dtc.predict(X_test))
print(f"Accuracy: {accuracy * 100:.2f}%\n\n")

#--------------------------------------------------------------------
# Step 6: Visualize the Tree
#--------------------------------------------------------------------
# Visualize the decision tree
class_names = [str(class_name) for class_name in pd.unique(wine['target'])]

graph = Source(tree.export_graphviz(dtc, out_file=None,
                                   feature_names=X.columns,
                                   class_names=class_names,
                                   filled=True))

# Render and display the decision tree
graph.render("decision_tree_wine")
display(graph)

#--------------------------------------------------------------------
# Step 7: Simulate Prediction
#--------------------------------------------------------------------
# Create a DataFrame with a simulated row of data to predict wine class
# Replace the values below with whatever you'd like to test
simulated_data = pd.DataFrame({
    "alcohol": [13.5],  # Example alcohol value
    "flavanoids": [2.8],  # Example flavanoids value
    "color_intensity": [5.0]  # Example color intensity value
})

# Use the trained decision tree classifier to predict the class of this simulated data
predicted_class = dtc.predict(simulated_data)

# Display the predicted class
print(f"Predicted class for the simulated data: {predicted_class[0]}")
