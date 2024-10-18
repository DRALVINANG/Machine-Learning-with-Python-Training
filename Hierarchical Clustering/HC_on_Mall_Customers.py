#---------------------------------------------------------------------------------------------
# Step 1: Import Required Libraries
#---------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

#---------------------------------------------------------------------------------------------
# Step 2: Load the Dataset
#---------------------------------------------------------------------------------------------
url = 'https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/main/Hierarchical%20Clustering/Mall%20Customers.csv'
df = pd.read_csv(url)

print("Original Data:\n", df.head())

#---------------------------------------------------------------------------------------------
# Step 3: Select Features for Clustering
#---------------------------------------------------------------------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

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

plt.axhline(y=6, color='r', linestyle='--')  # Set threshold for cluster separation
plt.title('Dendrogram for Customer Segmentation')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#---------------------------------------------------------------------------------------------
# Step 6: Perform Agglomerative Clustering
#---------------------------------------------------------------------------------------------
cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
df['Cluster'] = cluster.fit_predict(X_scaled)

print("\nClustered Data:\n", df.head())

#---------------------------------------------------------------------------------------------
# Step 7: Visualize the Clusters
#---------------------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
            c=df['Cluster'], cmap='viridis', s=100)
plt.title('Customer Segments Based on Hierarchical Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

#---------------------------------------------------------------------------------------------
# THE END
#---------------------------------------------------------------------------------------------
