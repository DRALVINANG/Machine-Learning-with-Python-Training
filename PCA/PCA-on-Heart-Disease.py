import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#--------------------------------------------------------------------
# Step 1: Load and Explore Dataset
#--------------------------------------------------------------------
# Load the heart disease dataset from the provided link
url = "https://github.com/DRALVINANG/Machine-Learning-with-Python-Training/blob/main/PCA/heart-disease.csv?raw=true"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

#--------------------------------------------------------------------
# Step 2: Data Preprocessing
#--------------------------------------------------------------------
# Drop any non-numeric columns if present and check for null values
df = df.dropna()  # Dropping rows with any missing values
df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns

# Separate features and target variable
X = df_numeric.drop(columns='target')  # Assuming 'target' is the column for diagnosis
y = df_numeric['target']

# Display the summary of the cleaned dataset
print(df_numeric.info())

#--------------------------------------------------------------------
# Step 3: Standardize the Data
#--------------------------------------------------------------------
# Standardize the data to have mean=0 and variance=1
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

#--------------------------------------------------------------------
# Step 4: Apply PCA with 10 Components
#--------------------------------------------------------------------
# Initialize PCA with 10 components
pca = PCA(n_components=10)

# Fit the PCA model to the scaled data
x_pca = pca.fit_transform(scaled_data)

#--------------------------------------------------------------------
# Step 5: Scree Plot (Variance Explained by Each Component)
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
# Step 6: Cumulative Variance Explained
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

# Print the variance explained by each principal component line by line
print("Variance explained by each component:")
for i, variance in enumerate(pca.explained_variance_ratio_, 1):
    print(f"Principal Component {i}: {variance:.6f}")

#--------------------------------------------------------------------
# Step 7: Apply PCA with 2 Components for Visualization
#--------------------------------------------------------------------
# Now, we will apply PCA with 2 components (since PC1 and PC2 are most important)
pca_2 = PCA(n_components=2)
x_pca_2 = pca_2.fit_transform(scaled_data)

#--------------------------------------------------------------------
# Step 8: Visualize Principal Components (Scatter Plot)
#--------------------------------------------------------------------
# Create a scatter plot for the first two principal components
plt.figure(figsize=(8, 6))

# Scatter plot with color representing the target variable (diagnosis)
scatter = plt.scatter(x_pca_2[:, 0], x_pca_2[:, 1], c=y, cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: First vs Second Principal Component')

# Create a legend for the target variable classes
legend1 = plt.legend(*scatter.legend_elements(), title="Target (Heart Disease)")
plt.gca().add_artist(legend1)

plt.show()

#--------------------------------------------------------------------
# Step 9: Get the Loadings (Principal Axes in Feature Space)
#--------------------------------------------------------------------
# Get the loadings (weights of features for PC1 and PC2)
loadings = pd.DataFrame(pca_2.components_.T, columns=['PC1', 'PC2'], index=X.columns)

# Display the loadings for each principal component
print("Feature Loadings for PC1 and PC2:\n", loadings)

#--------------------------------------------------------------------
# Step 10: Visualize the Loadings for PC1 and PC2
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

