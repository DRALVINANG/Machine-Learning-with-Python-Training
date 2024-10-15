# Install required libraries
!pip install plotly
!pip install pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Renaming columns for clarity to reflect annual spending in dollars
df = df.rename(columns={'Fresh': 'Annual Spending on Fresh Products ($)', 
                        'Milk': 'Annual Spending on Milk Products ($)', 
                        'Grocery': 'Annual Spending on Grocery Products ($)'})

# Creating 3D scatter plot with more informative labels and title
fig = px.scatter_3d(df, x='Annual Spending on Fresh Products ($)', 
                    y='Annual Spending on Milk Products ($)', 
                    z='Annual Spending on Grocery Products ($)',
                    color=df['Cluster'].astype(str),
                    title='Wholesale Customers Annual Purchasing Breakdown ($)',
                    labels={
                        'Annual Spending on Fresh Products ($)': 'Annual Fresh Products Spending ($)',
                        'Annual Spending on Milk Products ($)': 'Annual Milk Spending ($)',
                        'Annual Spending on Grocery Products ($)': 'Annual Grocery Spending ($)',
                    },
                    opacity=0.7)

# Show the plot in the notebook (rotatable and zoomable)
fig.update_layout(
    scene = dict(
        xaxis_title='Annual Fresh Products Spending ($)',
        yaxis_title='Annual Milk Spending ($)',
        zaxis_title='Annual Grocery Spending ($)'),
    title={
        'text': "Wholesale Customers' Purchasing Breakdown",
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    margin=dict(l=0, r=0, b=0, t=50),
)

fig.update_traces(marker=dict(size=4))  # Set size of dots
fig.show()

#--------------------------------------------------------------------
# Step 6: Print Cluster Insights
#--------------------------------------------------------------------
# Add color codes to the cluster insights
print("\nRevised Cluster Insights with Color Code:")
print("- **Cluster 0 (Purple)**: Low-value or infrequent buyers, spending small amounts on fresh products, milk, and grocery items.")
print("- **Cluster 1 (Red)**: Medium-value customers, spending moderate amounts on products.")
print("- **Cluster 2 (Green)**: High-value customers, spending large amounts on fresh products, milk, and grocery items.\n")

#--------------------------------------------------------------------
# Step 7: Predict New Data Points (Simulated Prediction)
#--------------------------------------------------------------------
# Function to predict which cluster a new customer would belong to
def predict_new_customer(fresh_spending, milk_spending, grocery_spending, frozen, detergents_paper, delicassen):
    # Standardize the new customer's data
    new_data_scaled = scaler.transform([[fresh_spending, milk_spending, grocery_spending, frozen, detergents_paper, delicassen]])
    # Predict the cluster
    cluster = kmeans.predict(new_data_scaled)[0]
    return cluster

# Simulate a new customer with sample data
new_customer = {
    'Annual Spending on Fresh Products ($)': 8000,
    'Annual Spending on Milk Products ($)': 1500,
    'Annual Spending on Grocery Products ($)': 3000,
    'Frozen': 2000,
    'Detergents_Paper': 600,
    'Delicassen': 1000
}
predicted_cluster = predict_new_customer(new_customer['Annual Spending on Fresh Products ($)'], 
                                         new_customer['Annual Spending on Milk Products ($)'], 
                                         new_customer['Annual Spending on Grocery Products ($)'],
                                         new_customer['Frozen'], 
                                         new_customer['Detergents_Paper'], 
                                         new_customer['Delicassen'])

# Print the simulated customer features line by line
print(f"Simulated New Customer Features:")
print(f" - Fresh Spending: {new_customer['Annual Spending on Fresh Products ($)']} $")
print(f" - Milk Spending: {new_customer['Annual Spending on Milk Products ($)']} $")
print(f" - Grocery Spending: {new_customer['Annual Spending on Grocery Products ($)']} $")
print(f" - Frozen: {new_customer['Frozen']} $")
print(f" - Detergents_Paper: {new_customer['Detergents_Paper']} $")
print(f" - Delicassen: {new_customer['Delicassen']} $")

# Display the predicted cluster for the new customer
print(f"\nThe new customer belongs to Cluster {predicted_cluster}.")
