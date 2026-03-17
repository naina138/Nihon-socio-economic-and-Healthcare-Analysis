import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("Japan_life_expectancy.csv")  

df_numeric = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)
# STEP 1: Elbow Method
inertia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.grid(True)
plt.tight_layout()
plt.savefig("inertia.png")
plt.show()

# STEP 2: Final KMeans with k=3

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# STEP 3: PCA for Visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
df['PC1'] = pca_components[:, 0]
df['PC2'] = pca_components[:, 1]

# PCA Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2')
plt.title('K-Means Clustering of Prefectures (k=3)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.savefig("c1c2.png")
plt.show()


# STEP 4: Cluster Profiling
cluster_profile = df.groupby('Cluster').mean(numeric_only=True)
print("\nCluster Profiles:\n", cluster_profile)
cluster_profile.to_csv("cluster_profiles.csv")
# STEP 5: Bar Chart of Cluster Means
profile_transposed = cluster_profile.T

profile_transposed.plot(kind='bar', figsize=(12, 6))
plt.title('Cluster-wise Feature Averages')
plt.xlabel('Features')
plt.ylabel('Mean Value (Standardized)')
plt.grid(True)
plt.tight_layout()
plt.savefig("cluster_barplot.png")
plt.show()
# STEP 6: Top Features for Each Cluster
for i in range(cluster_profile.shape[0]):
    print(f"\nTop Features for Cluster {i}:")
    print(cluster_profile.loc[i].sort_values(ascending=False).head(3))
    print("Least Contributing:")
    print(cluster_profile.loc[i].sort_values().head(3))

# STEP 7: Save Final Dataset with Cluster Labels

df.to_csv("clustered_data.csv", index=False)

