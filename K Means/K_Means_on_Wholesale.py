# Install required libraries
!pip install plotly
!pip install pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

#--------------------------------------------------------------------
# Step 1: Load the Wholesale Customers Dataset
#--------------------------------------------------------------------
# Load the dataset from UCI in CSV format
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

#--------------------------------------------------------------------
# Step 2: Data Preprocessing
#--------------------------------------------------------------------
# Drop rows with any missing values (if any)
df = df.dropna()

# Display dataset information to understand its structure
print(df.info())

# No CustomerID or non-numeric columns here, so we don't need additional filtering. We'll use all columns except the Channel and Region.
df_features = df.drop(columns=['Channel', 'Region'])

#--------------------------------------------------------------------
# Step 3: Standardize the Data
#--------------------------------------------------------------------
# Standardize the dataset using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

#--------------------------------------------------------------------
# Step 4: Apply K-Means and Find Optimal Clusters (Elbow Method)
#--------------------------------------------------------------------
# Use the elbow method to determine the optimal number of clusters
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.grid(True)
plt.show()

# Based on the Elbow Method, let's assume 3 clusters are optimal
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
display(df)

#--------------------------------------------------------------------
# Step 5: Visualize the Clusters in 3D (Rotatable)
#--------------------------------------------------------------------
# For visualization, we'll pick three dimensions from the dataset. Let's use 'Fresh', 'Milk', and 'Grocery' for the 3D plot.
fig = px.scatter_3d(df, x='Fresh', y='Milk', z='Grocery',
                    color=df['Cluster'].astype(str),
                    title='K-Means Clustering (3D Visualization)',
                    labels={'Fresh': 'Fresh', 'Milk': 'Milk', 'Grocery': 'Grocery'},
                    opacity=0.7)

# Show the plot in the notebook (rotatable and zoomable)
fig.update_traces(marker=dict(size=4))  # Set size of dots
fig.show()

#--------------------------------------------------------------------
# Step 6: Print Cluster Insights
#--------------------------------------------------------------------
print("\nRevised Cluster Insights:")
print("- **Cluster 0**: Low-value or infrequent buyers, purchasing small amounts of fresh, milk, and grocery items.")
print("- **Cluster 1**: Medium-value customers, buying moderate quantities of products.")
print("- **Cluster 2**: High-value customers, purchasing large quantities of fresh, milk, and grocery items.\n")

#--------------------------------------------------------------------
# Step 7: Predict New Data Points (Simulated Prediction)
#--------------------------------------------------------------------
# Function to predict which cluster a new customer would belong to
def predict_new_customer(fresh, milk, grocery, frozen, detergents_paper, delicassen):
    # Standardize the new customer's data
    new_data_scaled = scaler.transform([[fresh, milk, grocery, frozen, detergents_paper, delicassen]])
    # Predict the cluster
    cluster = kmeans.predict(new_data_scaled)[0]
    return cluster

# Simulate a new customer with sample data
new_customer = {
    'Fresh': 8000,
    'Milk': 1500,
    'Grocery': 3000,
    'Frozen': 2000,
    'Detergents_Paper': 600,
    'Delicassen': 1000
}
predicted_cluster = predict_new_customer(new_customer['Fresh'], new_customer['Milk'], new_customer['Grocery'],
                                         new_customer['Frozen'], new_customer['Detergents_Paper'], new_customer['Delicassen'])
print(f"Simulated New Customer Features: Fresh={new_customer['Fresh']}, Milk={new_customer['Milk']}, Grocery={new_customer['Grocery']}, Frozen={new_customer['Frozen']}, Detergents_Paper={new_customer['Detergents_Paper']}, Delicassen={new_customer['Delicassen']}")
print(f"The new customer belongs to Cluster {predicted_cluster}.")
