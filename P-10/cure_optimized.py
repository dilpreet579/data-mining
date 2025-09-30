import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scipy.spatial.distance import cdist

# ------------------------------
# 1. Dataset Preparation
# ------------------------------
# Synthetic dataset with irregular clusters and noise
X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

# ------------------------------
# 2. Implement CURE Algorithm
# ------------------------------

class CURE:
    def __init__(self, n_clusters=2, n_repr_points=5, shrink_factor=0.2):
        self.n_clusters = n_clusters
        self.n_repr_points = n_repr_points
        self.shrink_factor = shrink_factor
        self.clusters = []

    def fit(self, X):
        # Step 1: Initialize each point as a cluster
        self.clusters = [[i] for i in range(len(X))]

        # Step 2: Merge until desired clusters remain
        while len(self.clusters) > self.n_clusters:
            min_dist = float("inf")
            merge_a, merge_b = -1, -1

            # Find closest pair of clusters based on representative points
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    dist = self.cluster_distance(X, self.clusters[i], self.clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_a, merge_b = i, j

            # Merge the two clusters
            new_cluster = self.clusters[merge_a] + self.clusters[merge_b]
            self.clusters = [self.clusters[k] for k in range(len(self.clusters)) if k not in [merge_a, merge_b]]
            self.clusters.append(new_cluster)

        return self

    def cluster_distance(self, X, cluster_a, cluster_b):
        repr_a = self.representative_points(X, cluster_a)
        repr_b = self.representative_points(X, cluster_b)
        return np.min(cdist(repr_a, repr_b))

    def representative_points(self, X, cluster):
        points = X[cluster]
        centroid = np.mean(points, axis=0)

        # Select farthest points iteratively
        repr_points = []
        farthest = points[np.argmax(np.linalg.norm(points - centroid, axis=1))]
        repr_points.append(farthest)

        while len(repr_points) < min(self.n_repr_points, len(points)):
            dists = np.min(cdist(points, np.array(repr_points)), axis=1)
            new_point = points[np.argmax(dists)]
            repr_points.append(new_point)

        repr_points = np.array(repr_points)

        # Shrink towards centroid
        repr_points = repr_points + self.shrink_factor * (centroid - repr_points)
        return repr_points

    def predict(self, X):
        labels = np.zeros(len(X), dtype=int)
        for cluster_id, cluster in enumerate(self.clusters):
            for idx in cluster:
                labels[idx] = cluster_id
        return labels

# ------------------------------
# Run CURE
# ------------------------------
cure = CURE(n_clusters=2, n_repr_points=5, shrink_factor=0.2)
cure.fit(X)
labels = cure.predict(X)

# ------------------------------
# Plot Results
# ------------------------------
plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=30)
plt.title("CURE Clustering")
plt.show()
# ------------------------------
# Plot Results with Representative Points
# ------------------------------
plt.figure(figsize=(8,6))

# Plot clustered points
plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=30, alpha=0.6)

# Show representative points for each cluster
colors = ["red", "blue", "green", "orange", "purple"]
for cluster_id, cluster in enumerate(cure.clusters):
    points = X[cluster]
    centroid = np.mean(points, axis=0)

    # Original representative points (before shrinking)
    repr_points = []
    farthest = points[np.argmax(np.linalg.norm(points - centroid, axis=1))]
    repr_points.append(farthest)

    while len(repr_points) < min(cure.n_repr_points, len(points)):
        dists = np.min(cdist(points, np.array(repr_points)), axis=1)
        new_point = points[np.argmax(dists)]
        repr_points.append(new_point)

    repr_points = np.array(repr_points)

    # Shrunken representative points
    shrunk_points = repr_points + cure.shrink_factor * (centroid - repr_points)

    # Plot original representative points
    plt.scatter(repr_points[:,0], repr_points[:,1], marker='x', s=100,
                color=colors[cluster_id % len(colors)], label=f"Cluster {cluster_id} Repr.")

    # Plot shrunk representative points
    plt.scatter(shrunk_points[:,0], shrunk_points[:,1], marker='*', s=150,
                color=colors[cluster_id % len(colors)], edgecolor="black")

    # Draw shrinking arrows
    for orig, shrunk in zip(repr_points, shrunk_points):
        plt.arrow(orig[0], orig[1],
                  (shrunk[0] - orig[0]), (shrunk[1] - orig[1]),
                  color=colors[cluster_id % len(colors)], alpha=0.6,
                  head_width=0.03, length_includes_head=True)

plt.title("CURE Clustering with Representative Points and Shrinking")
plt.legend()
plt.show()