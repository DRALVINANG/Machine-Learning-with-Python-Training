#--------------------------------------------------------------------
# Step 1: Load Dataset and Define Features
#--------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
from sklearn import tree

# Load the Iris dataset
url = "https://www.alvinang.sg/s/iris_dataset.csv"
iris_df = pd.read_csv(url)

# Define the target and features
y = iris_df["species"]
X = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

#--------------------------------------------------------------------
# Step 2: Train Decision Tree Classifier
#--------------------------------------------------------------------
# Train the classifier using the entropy criterion
dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X, y)

#--------------------------------------------------------------------
# Step 3: Rank the Feature Importance Scores and Visualize
#--------------------------------------------------------------------
# Create a DataFrame for feature importance
feature_importances_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dtc.feature_importances_
})

# Sort by importance
feature_importances_df = feature_importances_df.sort_values(by="Importance", ascending=False)

# Plot with colors
plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importances_df, palette="Set2")

# Add labels and title
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visualizing Important Features using Decision Tree Classifier")

# Show the plot
plt.show()

#--------------------------------------------------------------------
# Step 4: Visualize the Decision Tree
#--------------------------------------------------------------------
# Visualize the decision tree using graphviz
graph = Source(tree.export_graphviz(dtc, out_file=None,
                                    feature_names=X.columns,
                                    class_names=pd.unique(iris_df["species"]),
                                    filled=True))
# Display the decision tree graph
graph.render("decision_tree", format="png", cleanup=False)
graph
