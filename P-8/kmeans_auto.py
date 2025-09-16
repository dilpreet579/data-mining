import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# -----------------------------
# SET DATASET
# -----------------------------
SAMPLES=700
CLUSTERS=4
STD=1.5

# -----------------------------
# Dataset Generation (make_blobs)
# -----------------------------
def generate_dataset(n_samples=300, n_clusters=4, cluster_std=0.8, random_state=42):
    X, true_labels = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X, true_labels

# -----------------------------
# Show Dataset
# -----------------------------
def plot_dataset(X):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], s=20, color="steelblue")
    plt.title("Generated Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.axis("equal")
    plt.show()

# -----------------------------
# K-Means from Scratch
# -----------------------------
def kmeans(X, k, max_iters=100, random_state=None):
    np.random.seed(random_state)
    # Step 1: Randomly pick initial centroids
    initial_idx = np.random.choice(len(X), k, replace=False)
    centroids = X[initial_idx]

    for _ in range(max_iters):
        # Step 2: Assign clusters
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Step 3: Recompute centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Convergence check
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels, centroids

# -----------------------------
# Plot Clusters
# -----------------------------
def plot_clusters(X, labels, centroids, title="K-Means Clustering Result"):
    plt.figure(figsize=(6, 6))
    for i in range(len(centroids)):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], s=20, label=f"Cluster {i+1}")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=200, label="Centroids")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.show()

# -----------------------------
# Elbow Method
# -----------------------------
def elbow_method(X, max_k=10):
    sse = []
    K = range(1, max_k + 1)
    for k in K:
        kmeans_model = KMeans(n_clusters=k, random_state=42).fit(X)
        sse.append(kmeans_model.inertia_)  # inertia = SSE
    plt.figure(figsize=(6, 6))
    plt.plot(K, sse, "o-", color="blue")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("SSE")
    plt.title("Elbow Method for Optimal k")
    plt.show()

# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    # Step 1: Generate dataset automatically with make_blobs
    X, true_labels = generate_dataset(n_samples=SAMPLES, n_clusters=CLUSTERS, cluster_std=STD, random_state=42)

    # Step 2: Show dataset
    plot_dataset(X)

    # Step 3: Run KMeans with different seeds
    labels, centroids = kmeans(X, k=4, random_state=1)
    plot_clusters(X, labels, centroids, "K-Means Clustering with k=4 (seed=1)")

    labels2, centroids2 = kmeans(X, k=4, random_state=10)
    plot_clusters(X, labels2, centroids2, "K-Means Clustering with k=4 (seed=10)")

    # Step 4: Elbow Method
    elbow_method(X, max_k=9)
