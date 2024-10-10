#--------------------------------------------------------------------
# Step 1: Load Dataset and Encode Labels
#--------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import OLS

# Read the Iris dataset into a Pandas DataFrame
iris_df = pd.read_csv("https://www.alvinang.sg/s/iris_dataset.csv")

# Create a LabelEncoder object for the target variable
le = LabelEncoder()
y = le.fit_transform(iris_df["species"])

# Define features
X = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

#--------------------------------------------------------------------
# Step 2: Fit OLS Model and Display Summary Report
#--------------------------------------------------------------------
model = OLS(y, X).fit()
print(model.summary())

#--------------------------------------------------------------------
# Step 3: Extract P-Values for Feature Importance
#--------------------------------------------------------------------
# Extract p-values from the model
p_values = pd.DataFrame((model.pvalues), columns=["P-value"]).sort_values(by="P-value", ascending=True)
print(p_values)

#--------------------------------------------------------------------
# Step 4: Combine DataFrame and Drop Unnecessary Columns
#--------------------------------------------------------------------
# Combine original dataset with the encoded target
combined_df = pd.concat([iris_df, pd.DataFrame({"target": y})], axis=1)
combined_df = combined_df.drop("species", axis=1)

#--------------------------------------------------------------------
# Step 5: Plot Regression Lines for Each Feature
#--------------------------------------------------------------------
# Create subplots for regression plots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Sepal Width
sns.regplot(x='target', y='sepal_width', data=combined_df, ax=axes[0, 0], color='red')
axes[0, 0].set_ylim(0,)
axes[0, 0].set_title('Sepal Width')

# Sepal Length
sns.regplot(x='target', y='sepal_length', data=combined_df, ax=axes[0, 1], color='green')
axes[0, 1].set_ylim(0,)
axes[0, 1].set_title('Sepal Length')

# Petal Length
sns.regplot(x='target', y='petal_length', data=combined_df, ax=axes[1, 0], color='orange')
axes[1, 0].set_ylim(0,)
axes[1, 0].set_title('Petal Length')

# Petal Width
sns.regplot(x='target', y='petal_width', data=combined_df, ax=axes[1, 1], color='blue')
axes[1, 1].set_ylim(0,)
axes[1, 1].set_title('Petal Width')

# Show the plot
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------
# Step 6: Visualize Feature Importance with Bar Plot
#--------------------------------------------------------------------
# Create a bar plot of 1 - P-value for feature importance
sns.barplot(x=1 - p_values['P-value'], y=p_values.index, palette='Set2')

# Add a red dashed line for the 0.95 threshold
plt.axvline(x=0.95, color='r', linestyle='dotted')

# Annotate the threshold
plt.annotate('0.95', xy=(0.95, 3.3), xycoords='data', color='r')

# Add labels and title
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features using Multiple Regression')

# Show the plot
plt.show()

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------
