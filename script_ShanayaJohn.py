from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


#Part 1: Selecting a dataset
#Source:https://archive.ics.uci.edu/dataset/257/user+knowledge+modeling
# Fetch dataset 
user_knowledge_modeling = fetch_ucirepo(id=257)

# Data (as pandas dataframes)
X = user_knowledge_modeling.data.features
y = user_knowledge_modeling.data.targets


#Part 2: Pre-processing the data

# Applying Transfrom: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# PCA analysis - reducing to 2 principal components
pca = PCA(n_components=2) 
reducedData = pca.fit_transform(X_scaled) #Applying transformation to the standarded data
reducedData= pd.DataFrame(reducedData, columns=['PC1', 'PC2'])

#Viewing the dataset before clustering 
plt.figure(figsize=(8, 6))
plt.scatter(reducedData['PC1'], reducedData['PC2'])
plt.title('User Knowledge Model')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

#Part 3: Clustering the data

#Elbow Method used to determine the optimal number of clusters
inertia = []

#Run possible k values
for k in range(1, 11): 
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reducedData)
    inertia.append(kmeans.inertia_)

# Plotting the inertia values(Elbow Curve)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

print("From the Elbow Method graph, the optimal k value is 3. ")


# Using the optimal k value to create a K-Mean model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(reducedData) # Fit the data to the K-Mean Model


# Printing out the cluster centroids
clusters = kmeans.cluster_centers_
print("Cluster Centers:\n", clusters)

# Predict the cluster labels
y_pred = kmeans.predict(reducedData)

# Add the predicted cluster labels to the reduced data
reducedData['Cluster'] = y_pred

# Plotting the data with the predicted labels using the two principal components (PC1 and PC2)
plt.figure(figsize=(8, 6))
plt.scatter(reducedData['PC1'], reducedData['PC2'], c=reducedData['Cluster'], cmap='viridis')
plt.title('User Knowledge Model Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
