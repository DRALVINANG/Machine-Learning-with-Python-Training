import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#--------------------------------------------------------------------
# Step 1: Load and Explore Dataset
#--------------------------------------------------------------------
# Load the breast cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Convert to a DataFrame for better visualization
df = pd.DataFrame(data['data'], columns=data['feature_names'])

# Display the first few rows of the dataset
print(df.head())

#--------------------------------------------------------------------
# Step 2: Standardize the Data
#--------------------------------------------------------------------
# Standardize the data to have mean=0 and variance=1
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#--------------------------------------------------------------------
# Step 3: Apply PCA with 10 Components
#--------------------------------------------------------------------
# Initialize PCA with 10 components
pca = PCA(n_components=10)

# Fit the PCA model to the scaled data
x_pca = pca.fit_transform(scaled_data)

#--------------------------------------------------------------------
# Step 4: Scree Plot (Variance Explained by Each Component)
#--------------------------------------------------------------------
# Generate a scree plot to visualize explained variance
plt.figure(figsize=(8, 6))
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot: Variance Explained by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

#--------------------------------------------------------------------
# Step 5: Cumulative Variance Explained
#--------------------------------------------------------------------
# Calculate cumulative variance explained by the components
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative variance to show elbow effect
plt.figure(figsize=(8, 6))
plt.plot(PC_values, cumulative_variance, marker='o', linestyle='--', color='green')
plt.title('Cumulative Variance Explained by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance Explained')
plt.show()

# Print the variance explained by each principal component
print("Variance explained by each component:\n", pca.explained_variance_ratio_)

#--------------------------------------------------------------------
# Step 6: Apply PCA with 2 Components for Visualization
#--------------------------------------------------------------------
# Now, we will apply PCA with 2 components (since PC1 and PC2 are most important)
pca_2 = PCA(n_components=2)
x_pca_2 = pca_2.fit_transform(scaled_data)

#--------------------------------------------------------------------
# Step 7: Visualize Principal Components (Scatter Plot)
#--------------------------------------------------------------------
# Create a scatter plot for the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(x_pca_2[:, 0], x_pca_2[:, 1], c=data['target'], cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: First vs Second Principal Component')
plt.show()

#--------------------------------------------------------------------
# Step 8: Get the Loadings (Principal Axes in Feature Space)
#--------------------------------------------------------------------
# Get the loadings (weights of features for PC1 and PC2)
loadings = pd.DataFrame(pca_2.components_.T, columns=['PC1', 'PC2'], index=df.columns)

# Display the loadings for each principal component
print("Feature Loadings for PC1 and PC2:\n", loadings)

#--------------------------------------------------------------------
# Step 9: Visualize the Loadings for PC1 and PC2
#--------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(loadings.index, loadings['PC1'], color='blue', label='PC1')
plt.bar(loadings.index, loadings['PC2'], color='green', bottom=loadings['PC1'], label='PC2')
plt.xticks(rotation=90)
plt.ylabel('Loading Value')
plt.title('Feature Loadings for PC1 and PC2')
plt.legend()
plt.show()

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------
