# Install plotly if not already installed
!pip install plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px  # For interactive 3D plotting

#--------------------------------------------------------------------
# Step 1: Load and Explore Dataset
#--------------------------------------------------------------------
# Load the heart disease dataset from the provided link
url = "https://github.com/DRALVINANG/Machine-Learning-with-Python-Training/blob/main/PCA/heart-disease.csv?raw=true"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

#--------------------------------------------------------------------
# Step 2: Handle 'ca' and 'thal' Columns (Convert to Numeric)
#--------------------------------------------------------------------
# Replace '?' with NaN for 'ca' and 'thal' columns
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')  # Convert to numeric, coerce errors
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')  # Convert to numeric, coerce errors

# Drop rows with NaN values
df.dropna(inplace=True)

# Verify that 'ca' and 'thal' are now numeric and no NaN values remain
print(df[['ca', 'thal']].info())

#--------------------------------------------------------------------
# Step 3: Separate Features and Target Variable
#--------------------------------------------------------------------
# Separate features and target variable
X = df.drop(columns='target')  # Assuming 'target' is the column for diagnosis
y = df['target']

# Display the summary of the cleaned dataset
print(df.info())

#--------------------------------------------------------------------
# Step 4: Standardize the Data
#--------------------------------------------------------------------
# Standardize the data to have mean=0 and variance=1
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

#--------------------------------------------------------------------
# Step 5: Apply PCA with 10 Components
#--------------------------------------------------------------------
# Initialize PCA with 10 components
pca = PCA(n_components=10)

# Fit the PCA model to the scaled data
x_pca = pca.fit_transform(scaled_data)

#--------------------------------------------------------------------
# Step 6: Scree Plot (Variance Explained by Each Component)
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
# Step 7: Cumulative Variance Explained
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
# Step 8: Apply PCA with 3 Components for Visualization
#--------------------------------------------------------------------
# Apply PCA with 3 components (since we want to visualize and analyze PC1, PC2, PC3)
pca_3 = PCA(n_components=3)
x_pca_3 = pca_3.fit_transform(scaled_data)

#--------------------------------------------------------------------
# Step 9: Visualize Principal Components (3D Scatter Plot with Plotly)
#--------------------------------------------------------------------
# Create an interactive 3D scatter plot using Plotly with smaller points
fig = px.scatter_3d(
    x=x_pca_3[:, 0],
    y=x_pca_3[:, 1],
    z=x_pca_3[:, 2],
    color=y,  # Color the points by the 'target' variable
    labels={'x': 'First Principal Component', 'y': 'Second Principal Component', 'z': 'Third Principal Component'},
    title='PCA: First vs Second vs Third Principal Component',
    opacity=0.7  # Adjust the opacity if needed for better visibility
)

# Show the plot in the notebook (rotatable and zoomable) with smaller points
fig.update_traces(marker=dict(size=3))  # Set size of dots to be smaller
fig.show()

#--------------------------------------------------------------------
# Step 10: Get the Loadings (Principal Axes in Feature Space)
#--------------------------------------------------------------------
# Get the loadings (weights of features for PC1, PC2, and PC3)
loadings = pd.DataFrame(pca_3.components_.T, columns=['PC1', 'PC2', 'PC3'], index=X.columns)

# Display the loadings for each principal component
print("Feature Loadings for PC1, PC2, and PC3:\n", loadings)

#--------------------------------------------------------------------
# Step 11: Visualize the Loadings for PC1, PC2, and PC3
#--------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(loadings.index, loadings['PC1'], color='blue', label='PC1')
plt.bar(loadings.index, loadings['PC2'], color='green', bottom=loadings['PC1'], label='PC2')
plt.bar(loadings.index, loadings['PC3'], color='orange', bottom=loadings['PC1']+loadings['PC2'], label='PC3')
plt.xticks(rotation=90)
plt.ylabel('Loading Value')
plt.title('Feature Loadings for PC1, PC2, and PC3')
plt.legend()
plt.show()

#--------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------
