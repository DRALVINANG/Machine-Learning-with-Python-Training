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
# Step 1: Load the Online Retail Dataset
#--------------------------------------------------------------------
# Load the dataset from UCI in Excel format
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
df = pd.read_excel(url)

# Display the first few rows of the dataset
print(df.head())

#--------------------------------------------------------------------
# Step 2: Data Preprocessing
#--------------------------------------------------------------------
# Drop rows with missing CustomerID and Description
df = df.dropna(subset=['CustomerID', 'Description'])

# Remove any negative or zero quantities, as they likely represent returns or errors
df = df[df['Quantity'] > 0]

# Create a new feature for total sales (Quantity * UnitPrice)
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# Group by CustomerID to get aggregated features for clustering
customer_df = df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalSales': 'sum',
    'InvoiceNo': 'nunique'
}).reset_index()

# Rename columns
customer_df.columns = ['CustomerID', 'TotalQuantity', 'TotalSales', 'UniqueInvoices']
display(customer_df)

#--------------------------------------------------------------------
# Step 3: Standardize the Data
#--------------------------------------------------------------------
# Standardize the dataset using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_df[['TotalQuantity', 'TotalSales', 'UniqueInvoices']])

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
customer_df['Cluster'] = kmeans.fit_predict(X_scaled)
display(customer_df)

#--------------------------------------------------------------------
# Step 5: Visualize the Clusters in 3D (Rotatable)
#--------------------------------------------------------------------
# Create a 3D plot using Plotly for better visualization
fig = px.scatter_3d(customer_df, x='TotalQuantity', y='TotalSales', z='UniqueInvoices',
                    color=customer_df['Cluster'].astype(str),
                    title='K-Means Clustering (3D Visualization)',
                    labels={'TotalQuantity': 'Total Quantity', 'TotalSales': 'Total Sales', 'UniqueInvoices': 'Unique Invoices'},
                    opacity=0.7)

# Show the plot in the notebook (rotatable and zoomable)
fig.update_traces(marker=dict(size=4))  # Set size of dots
fig.show()

#--------------------------------------------------------------------
# Step 6: Print Cluster Insights
#--------------------------------------------------------------------
print("\nRevised Cluster Insights:")
print("- **Cluster 0** (Red): Low-value or infrequent buyers, with low total sales, low quantities, and fewer invoices.")
print("- **Cluster 2** (Purple): Medium-value customers, buying moderately but regularly.")
print("- **Cluster 1** (Green): High-value customers, purchasing large quantities with high total sales.\n")

#--------------------------------------------------------------------
# Step 7: Predict New Data Points (Simulated Prediction)
#--------------------------------------------------------------------
# Function to predict which cluster a new customer would belong to
def predict_new_customer(total_quantity, total_sales, unique_invoices):
    # Standardize the new customer's data
    new_data_scaled = scaler.transform([[total_quantity, total_sales, unique_invoices]])
    # Predict the cluster
    cluster = kmeans.predict(new_data_scaled)[0]
    return cluster

# Simulate a new customer with sample data
new_customer = {
    'TotalQuantity': 500,
    'TotalSales': 12000,
    'UniqueInvoices': 5
}
predicted_cluster = predict_new_customer(new_customer['TotalQuantity'], new_customer['TotalSales'], new_customer['UniqueInvoices'])
print(f"Simulated New Customer Features: TotalQuantity={new_customer['TotalQuantity']}, TotalSales={new_customer['TotalSales']}, UniqueInvoices={new_customer['UniqueInvoices']}")
print(f"The new customer belongs to Cluster {predicted_cluster}.")
