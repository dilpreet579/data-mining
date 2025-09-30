import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scipy.spatial.distance import cdist


class CURE:
    def __init__(self, n_clusters=2, n_representatives=10, shrink_factor=0.2):
        self.n_clusters = n_clusters
        self.n_representatives = n_representatives
        self.shrink_factor = shrink_factor

    def fit(self, X):
        clusters = [[i] for i in range(len(X))]

        while len(clusters) > self.n_clusters:
            print(f"{len(clusters)}")
            min_dist = float('inf')
            to_merge = (None, None)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self.cluster_distance(X, clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (i, j)

            i, j = to_merge
            clusters[i] += clusters[j]
            del clusters[j]

        final_clusters = []
        for cluster in clusters:
            points = X[cluster]
            rep_points = self.select_representatives(points)
            centroid = np.mean(points, axis=0)
            rep_points = centroid + self.shrink_factor * (rep_points - centroid)
            final_clusters.append((points, rep_points))

        self.labels_ = np.zeros(len(X), dtype=int)
        for idx, (points, _) in enumerate(final_clusters):
            for p in points:
                self.labels_[np.where((X == p).all(axis=1))[0][0]] = idx

        self.final_clusters = final_clusters
        return self

    def select_representatives(self, points):
        """ Spread representatives across the cluster using farthest-point sampling """
        centroid = np.mean(points, axis=0)
        # Start with farthest point from centroid
        rep_points = [points[np.argmax(np.linalg.norm(points - centroid, axis=1))]]

        for _ in range(self.n_representatives - 1):
            # Pick point farthest from ALL already chosen reps
            dists = np.min(cdist(points, np.array(rep_points)), axis=1)
            rep_points.append(points[np.argmax(dists)])

        return np.array(rep_points)

    def cluster_distance(self, X, cluster1, cluster2):
        points1 = X[cluster1]
        points2 = X[cluster2]
        reps1 = self.select_representatives(points1)
        reps2 = self.select_representatives(points2)
        return np.min(cdist(reps1, reps2))


# ---- MAIN ----
X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

cure = CURE(n_clusters=2, n_representatives=12, shrink_factor=0.15)
cure.fit(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cure.labels_, cmap='viridis', s=30, alpha=0.7)

for _, reps in cure.final_clusters:
    plt.scatter(reps[:, 0], reps[:, 1], c='red', marker='x', s=100)

plt.title("CURE Clustering (Improved representative point spread)")
plt.show()
