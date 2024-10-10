#--------------------------------------------------------------------
# Step 1: Load Dataset and Define Features
#--------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import graphviz
from sklearn.tree import export_graphviz

# Load the Iris dataset into a Pandas DataFrame
url = "https://www.alvinang.sg/s/iris_dataset.csv"
iris_df = pd.read_csv(url)

# Define the target and features
y = iris_df["species"]
X = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

#--------------------------------------------------------------------
# Step 2: Train Random Forest Classifier
#--------------------------------------------------------------------
# Create a RandomForestClassifier object and fit the model to the data
rfc = RandomForestClassifier(n_estimators=3, criterion="entropy", random_state=42)
rfc.fit(X, y)

#--------------------------------------------------------------------
# Step 3: Rank the Feature Importance Scores and Visualize
#--------------------------------------------------------------------
# Get the feature importances
feature_importances_rfc = rfc.feature_importances_

# Create a Pandas DataFrame for feature importance
feature_importances_df_rfc = pd.DataFrame(
    data={"feature": X.columns, "importance": feature_importances_rfc}
)

# Sort the DataFrame by importance
feature_importances_df_rfc = feature_importances_df_rfc.sort_values(by="importance", ascending=False)

# Print the feature importance DataFrame
print(feature_importances_df_rfc)

# Plot the feature importance with colors
plt.figure(figsize=(8, 6))
sns.barplot(x="importance", y="feature", data=feature_importances_df_rfc, palette="Set2")

# Add labels and title
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features using Random Forest Classifier')

# Show the plot
plt.show()

#--------------------------------------------------------------------
# Step 4: Visualize the Trees in the Random Forest
#--------------------------------------------------------------------
# Extract individual decision trees from the Random Forest
estimators = rfc.estimators_

# Visualize each tree using Graphviz
for i, estimator in enumerate(estimators):
    dot_data = export_graphviz(estimator, out_file=None,
                               feature_names=X.columns,
                               class_names=["Setosa", "Versicolor", "Virginica"],
                               filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    # Display each tree inline in the notebook
    display(graph)

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------
