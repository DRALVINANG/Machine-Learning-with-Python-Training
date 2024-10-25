#---------------------------------------------------------------------------------------------
# Step 1: Install and Import Required Libraries
#---------------------------------------------------------------------------------------------
!pip install plotly  # Ensure Plotly is installed

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import plotly.express as px

#---------------------------------------------------------------------------------------------
# Step 2: Load the Dataset
#---------------------------------------------------------------------------------------------
url = 'https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/main/Hierarchical%20Clustering/Mall%20Customers.csv'
df = pd.read_csv(url)

print("Original Data:\n", df.head())

#---------------------------------------------------------------------------------------------
# Step 3: Select Features for Clustering
#---------------------------------------------------------------------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]

print("\nSelected Features for Clustering:\n", X.head())

#---------------------------------------------------------------------------------------------
# Step 4: Standardize the Data
#---------------------------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data:\n", X_scaled[:5])

#---------------------------------------------------------------------------------------------
# Step 5: Generate the Dendrogram
#---------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.axhline(y=7, color='r', linestyle='--')  # Adjusted threshold to suggest 6 clusters
plt.title('Dendrogram for Customer Segmentation')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#---------------------------------------------------------------------------------------------
# Step 6: Perform Agglomerative Clustering with 6 Clusters
#---------------------------------------------------------------------------------------------
cluster = AgglomerativeClustering(n_clusters=6, metric='euclidean', linkage='ward')
df['Cluster'] = cluster.fit_predict(X_scaled)

print("\nClustered Data:\n", df.head())

#---------------------------------------------------------------------------------------------
# Step 7: Visualize the Clusters in Interactive 3D using Plotly
#---------------------------------------------------------------------------------------------
fig = px.scatter_3d(df, 
                    x='Annual Income (k$)', 
                    y='Spending Score (1-100)', 
                    z='Age', 
                    color='Cluster', 
                    title='Customer Segments Based on Hierarchical Clustering (3D)',
                    symbol='Cluster', 
                    size_max=10, 
                    opacity=0.8)

fig.show()
