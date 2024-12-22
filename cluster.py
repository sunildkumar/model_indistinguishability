import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm


def load_vectors(fname="model_eval_results.csv"):
    """
    Loads prediction vectors from file. Returns a matrix of shape (10000, 5) - (num_examples, num_models)
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File {fname} does not exist")

    df = pd.read_csv(fname)
    # Extract relevant columns and convert to NumPy array directly
    vectors = df[
        ["ViT Label", "ConvNext Label", "Swin Label", "ViT16Lora Label", "ResNet Label"]
    ].to_numpy(dtype=int)

    return vectors


def multiclass_chebyshev_distance(vectors, cache_file="dist_matrix.npy"):
    """
    Compute the pairwise multiclass Chebyshev distance matrix using GPU acceleration.
    Cache the result to disk to avoid recomputation.
    """
    # Check if the cached file exists
    if os.path.exists(cache_file):
        print(f"Loading distance matrix from {cache_file}")
        return np.load(cache_file)

    # Transfer vectors to GPU
    vectors_gpu = cp.array(vectors)

    # Vectorized comparison using broadcasting on GPU
    def chebyshev_metric_gpu(v1, v2):
        return cp.max(v1 != v2, axis=-1).astype(int)

    # Compute pairwise distances on GPU
    dist_matrix_gpu = cp.zeros((vectors_gpu.shape[0], vectors_gpu.shape[0]), dtype=int)
    for i in tqdm(range(vectors_gpu.shape[0])):
        dist_matrix_gpu[i] = chebyshev_metric_gpu(vectors_gpu[i], vectors_gpu)

    # Transfer the result back to CPU
    dist_matrix = cp.asnumpy(dist_matrix_gpu)

    # Save the computed distance matrix to disk
    np.save(cache_file, dist_matrix)
    print(f"Distance matrix saved to {cache_file}")

    return dist_matrix


def cluster_with_kmedoids(dist_matrix, n_clusters=2):
    """
    Cluster the data using K-Medoids with the precomputed distance matrix.
    """
    print("Clustering with K-Medoids...")
    kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=42)
    cluster_labels = kmedoids.fit_predict(dist_matrix)

    print(f"Cluster centers (medoid indices): {kmedoids.medoid_indices_}")
    print(f"Cluster labels: {np.bincount(cluster_labels)}")

    return cluster_labels, kmedoids


def visualize_distance_matrix(
    dist_matrix, filename="distance_matrix.png", fraction=0.1
):
    """
    Visualize a fraction of the distance matrix as a heatmap.
    """
    # Determine the number of samples based on the fraction
    sample_size = int(dist_matrix.shape[0] * fraction)

    # Ensure at least one sample is selected
    sample_size = max(1, sample_size)

    # Randomly select indices for the sample
    indices = np.random.choice(dist_matrix.shape[0], sample_size, replace=False)
    sampled_matrix = dist_matrix[np.ix_(indices, indices)]

    print(f"Sampled matrix shape: {sampled_matrix.shape}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(sampled_matrix, cmap="viridis", cbar=True)
    plt.title("Distance Matrix Heatmap (Sampled)")
    plt.xlabel("Data Point Index")
    plt.ylabel("Data Point Index")
    plt.savefig(filename)
    plt.close()
    print(f"Distance matrix heatmap saved to {filename}")


def main():
    # Load vectors and compute the optimized distance matrix
    vectors = load_vectors()
    print("vectors loaded")
    dist_matrix = multiclass_chebyshev_distance(vectors)
    print("distance matrix loaded")

    # Visualize the distance matrix
    visualize_distance_matrix(dist_matrix)

    # Cluster the data using K-Medoids
    cluster_labels, kmedoids = cluster_with_kmedoids(dist_matrix, n_clusters=4)
    print("Clustering completed")
    print(f"Number of examples in each cluster: {np.bincount(cluster_labels)}")


if __name__ == "__main__":
    main()
