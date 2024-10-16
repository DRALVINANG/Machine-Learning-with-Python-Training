import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

#---------------------------------------------------------------------------------------------
# Step 1: Load the dataset
#---------------------------------------------------------------------------------------------
data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Spending_Groceries': [1000, 1500, 700, 3000, 2000],
    'Spending_Clothes': [200, 400, 150, 600, 500],
    'Spending_Electronics': [300, 800, 250, 900, 1000]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

#---------------------------------------------------------------------------------------------
# Step 2: Standardize the Data
#---------------------------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Spending_Groceries', 
                                    'Spending_Clothes', 
                                    'Spending_Electronics']])
print("\nScaled Data:\n", X_scaled)

#---------------------------------------------------------------------------------------------
# Step 3: Generate the Dendrogram
#---------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))

plt.axhline(y=2, color='r', linestyle='--')  
# Highlight the cluster cut point 
# (set based on your interpretation)

plt.title('Dendrogram for Customer Segmentation')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#---------------------------------------------------------------------------------------------
# Step 4: Agglomerative Clustering (Hierarchical Clustering)
#---------------------------------------------------------------------------------------------
cluster = AgglomerativeClustering(n_clusters=3, 
                                  metric='euclidean', 
                                  linkage='ward')
df['Cluster'] = cluster.fit_predict(X_scaled)

print("\nClustered Data:\n", df)

#---------------------------------------------------------------------------------------------
# THE END
#---------------------------------------------------------------------------------------------
