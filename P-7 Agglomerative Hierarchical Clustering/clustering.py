import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# VARIABLES FOR DATA GENERATION
DATAPOINTS = 70
GROUPS = 3
SCATTERING=2.5 # default = 1 for cluster_std in make_blobs
# cluster_std=0.5 → very tight blobs.
# cluster_std=1.0 (default) → medium scatter.
# cluster_std=2.5 → very loose blobs, overlapping more.
# With cluster_std=4.0, the dataset may look almost like random noise.

CLUSTERS = 5  # the end clusters to stop at

# -----------------------------
# 1. Dataset Generation
# -----------------------------
def generate_dataset(n_samples=70, n_clusters=3, cluster_std=1, random_state=42):
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters,
                      n_features=2, cluster_std=cluster_std,
                      random_state=random_state)
    return X


# -----------------------------
# 2. Distance Calculation
# -----------------------------
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def cluster_distance(c1, c2, linkage="single"):
    distances = [euclidean_distance(p1, p2) for p1 in c1 for p2 in c2]
    
    if linkage == "single":
        return np.min(distances)
    elif linkage == "complete":
        return np.max(distances)
    elif linkage == "average":
        return np.mean(distances)
    else:
        raise ValueError("Invalid linkage type!")

# -----------------------------
# 3. Agglomerative Hierarchical Clustering
# -----------------------------
def agglomerative_clustering(X, linkage="single", k=3):
    # Start with each point as its own cluster
    clusters = [[x] for x in X]
    
    while len(clusters) > k:  # <-- stop when we have k clusters
        min_dist = float("inf")
        pair_to_merge = None
        
        # Find two closest clusters
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = cluster_distance(clusters[i], clusters[j], linkage)
                if dist < min_dist:
                    min_dist = dist
                    pair_to_merge = (i, j)
        
        # Merge the closest clusters
        i, j = pair_to_merge
        new_cluster = clusters[i] + clusters[j]
        
        # Update cluster list
        clusters = [clusters[m] for m in range(len(clusters)) if m not in (i, j)]
        clusters.append(new_cluster)
    
    return clusters

# -----------------------------
# 4. Visualization
# -----------------------------
def plot_raw_dataset(X):
    plt.figure(figsize=(6,5))
    plt.scatter(X[:,0], X[:,1], c="black", marker="o")
    plt.title("Generated Dataset (before clustering)")
    plt.show()

def plot_clusters(clusters, title):
    colors = ["red", "blue", "green", "purple", "orange", "brown"]
    plt.figure(figsize=(6,5))
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:,0], cluster[:,1],
                    color=colors[i % len(colors)], label=f"Cluster {i+1}")
    plt.title(title)
    plt.legend()
    plt.show()

# -----------------------------
# Run Demo
# -----------------------------
if __name__ == "__main__":
    # Step 1: Generate dataset
    X = generate_dataset(n_samples=DATAPOINTS, n_clusters=GROUPS, cluster_std=SCATTERING)
    # cluster_std = 1 (default standard deviation)
    
    # Show raw dataset
    plot_raw_dataset(X)

    # Step 2: Run clustering with k=3
    for linkage in ["single", "complete", "average"]:
        final_clusters = agglomerative_clustering(X, linkage=linkage, k=CLUSTERS)
        plot_clusters(final_clusters, f"Agglomerative Clustering ({linkage}-linkage, k={CLUSTERS})")
