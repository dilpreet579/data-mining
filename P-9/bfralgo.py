import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# -----------------------------
# 1. Dataset Preparation
# -----------------------------
# Generate a large synthetic dataset (e.g., 1 million points in 5D)
n_samples = 1_000_00  # 100k points
n_features = 5
n_clusters = 5  # desired number of clusters
chunk_size = 20000  # how many points per chunk

X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Split dataset into chunks
chunks = [X[i:i+chunk_size] for i in range(0, len(X), chunk_size)]

# -----------------------------
# 2. Initial Cluster Formation (on first chunk)
# -----------------------------
first_chunk = chunks[0]

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(first_chunk)
centroids = kmeans.cluster_centers_

# Cluster statistics:
cluster_stats = {
    i: {
        'N': 0,
        'SUM': np.zeros(n_features),
        'SUMSQ': np.zeros(n_features)
    } for i in range(n_clusters)
}

# Fill stats for first chunk
for i, point in enumerate(first_chunk):
    c = labels[i]
    cluster_stats[c]['N'] += 1
    cluster_stats[c]['SUM'] += point
    cluster_stats[c]['SUMSQ'] += point ** 2

# -----------------------------
# 3. Incremental Processing of Chunks
# -----------------------------
def assign_point_to_cluster(point, centroids):
    """Assign a single point to the nearest centroid."""
    dists = np.linalg.norm(centroids - point, axis=1)
    return np.argmin(dists)

# Process subsequent chunks
for chunk in chunks[1:]:
    for point in chunk:
        c = assign_point_to_cluster(point, centroids)
        cluster_stats[c]['N'] += 1
        cluster_stats[c]['SUM'] += point
        cluster_stats[c]['SUMSQ'] += point ** 2

    # -----------------------------
    # 4. Centroid Recalculation
    # -----------------------------
    for c in range(n_clusters):
        if cluster_stats[c]['N'] > 0:
            centroids[c] = cluster_stats[c]['SUM'] / cluster_stats[c]['N']

# -----------------------------
# 5. Final Cluster Evaluation
# -----------------------------
print("Final Cluster Centroids after all chunks are processed:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i}: {centroid}")

print("\nCluster Size Distribution (number of points per cluster):")
for i in range(n_clusters):
    print(f"Cluster {i}: {cluster_stats[i]['N']} points")
