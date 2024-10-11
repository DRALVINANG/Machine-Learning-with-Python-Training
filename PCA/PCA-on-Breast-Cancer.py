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

# Convert scaled data back to DataFrame for easier reference
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print("Standardized data (first 5 rows):\n", scaled_df.head())

#--------------------------------------------------------------------
# Step 3: Apply PCA (Principal Component Analysis)
#--------------------------------------------------------------------
# Initialize PCA, specifying number of components
pca = PCA(n_components=2)

# Fit the PCA model to the scaled data and transform it
x_pca = pca.fit_transform(scaled_data)

# Display the shape of the reduced data
print("Shape of data after PCA: ", x_pca.shape)

#--------------------------------------------------------------------
# Step 4: Visualize Principal Components
#--------------------------------------------------------------------
# Create a scatter plot for the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=data['target'], cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: First vs Second Principal Component')
plt.show()

#--------------------------------------------------------------------
# Step 5: Scree Plot (Variance Explained by Each Component)
#--------------------------------------------------------------------
# Generate a scree plot to see how much variance is explained by each principal component
plt.figure(figsize=(8, 6))
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot: Variance Explained by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

#--------------------------------------------------------------------
# Step 6: Analyze Variance Explained by Principal Components
#--------------------------------------------------------------------
# Print the amount of variance explained by each of the principal components
explained_variance = pca.explained_variance_ratio_
print("Variance explained by each component: ", explained_variance)

#--------------------------------------------------------------------
# Step 7: Cumulative Variance Explained
#--------------------------------------------------------------------
# Calculate cumulative variance explained by the components
cumulative_variance = np.cumsum(explained_variance)

# Plot cumulative variance
plt.figure(figsize=(8, 6))
plt.plot(PC_values, cumulative_variance, marker='o', linestyle='--', color='green')
plt.title('Cumulative Variance Explained by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance Explained')
plt.show()

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------
