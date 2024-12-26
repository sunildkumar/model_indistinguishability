import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def multiclass_distance_matrix(
    vectors, metric="chebyshev", cache_file_prefix="dist_matrix"
):
    """
    Compute the pairwise multiclass distance matrix using GPU acceleration.
    Supports 'chebyshev' and 'hamming' metrics. Cache the result to disk to avoid recomputation.
    """
    # Determine the cache file name based on the metric
    cache_file = f"{cache_file_prefix}_{metric}.npy"

    # Check if the cached file exists
    if os.path.exists(cache_file):
        print(f"Loading distance matrix from {cache_file}")
        return np.load(cache_file)

    # Transfer vectors to GPU
    vectors_gpu = cp.array(vectors)

    # Define the metric functions
    def chebyshev_metric_gpu(v1, v2):
        return cp.max(v1 != v2, axis=-1).astype(int)

    def hamming_metric_gpu(v1, v2):
        return cp.mean(v1 != v2, axis=-1).astype(float)

    # Select the appropriate metric function
    if metric == "chebyshev":
        metric_function = chebyshev_metric_gpu
    elif metric == "hamming":
        metric_function = hamming_metric_gpu
    else:
        raise ValueError("Unsupported metric. Choose 'chebyshev' or 'hamming'.")

    # Compute pairwise distances on GPU
    dist_matrix_gpu = cp.zeros(
        (vectors_gpu.shape[0], vectors_gpu.shape[0]), dtype=float
    )
    for i in tqdm(range(vectors_gpu.shape[0])):
        dist_matrix_gpu[i] = metric_function(vectors_gpu[i], vectors_gpu)

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
    kmedoids = KMedoids(
        n_clusters=n_clusters, metric="precomputed", random_state=42, init="k-medoids++"
    )
    cluster_labels = kmedoids.fit_predict(dist_matrix)

    print(f"Cluster centers (medoid indices): {kmedoids.medoid_indices_}")
    print(f"Cluster labels: {np.bincount(cluster_labels)}")

    return cluster_labels, kmedoids


def save_clustered_dataframe(
    *,
    vectors,
    cluster_labels,
    original_file="model_eval_results.csv",
    output_file="model_eval_results_with_clusters.csv",
):
    """
    Save the original dataframe with an additional column for cluster labels.
    """
    # Load the original dataframe
    df = pd.read_csv(original_file)

    # Add the cluster labels to the dataframe
    df["Cluster Label"] = cluster_labels

    # Save the new dataframe with cluster labels
    df.to_csv(output_file, index=False)
    print(f"Dataframe with cluster labels saved to {output_file}")


def plot_per_cluster_accuracy(
    clustered_file="model_eval_results_with_clusters.csv",
    human_file="cifar10h-raw.csv",
    metric="chebyshev",
):
    """
    Plot the accuracy of each model and humans in each cluster and the size of each cluster.
    """
    df = pd.read_csv(clustered_file)
    human_df = pd.read_csv(human_file)
    cluster_indices = sorted(df["Cluster Label"].unique())

    # accuracies[cluster_index][model_name] = accuracy
    accuracies = {}
    cluster_sizes = {}

    for cluster in cluster_indices:
        accuracies[cluster] = {}
        cluster_df = df[df["Cluster Label"] == cluster]
        cluster_sizes[cluster] = len(cluster_df)

        # Select rows from human_df where image_filename matches filenames in cluster_df
        matching_human_df = human_df[
            human_df["image_filename"].isin(cluster_df["filename"])
        ]

        for model in ["ViT", "ConvNext", "Swin", "ViT16Lora", "ResNet"]:
            model_df = cluster_df[f"{model} Label"]
            accuracy = (model_df == cluster_df["True label"]).mean()
            accuracies[cluster][model] = accuracy

        human_accuracy = matching_human_df["correct_guess"].mean()
        accuracies[cluster]["Human"] = human_accuracy

    # Create a figure with two subplots
    fig, ax1 = plt.subplots(2, 1, figsize=(10, 12))

    # Plot the accuracies
    models = ["ViT", "ConvNext", "Swin", "ViT16Lora", "ResNet", "Human"]
    cluster_labels = [f"Cluster {cluster}" for cluster in cluster_indices]
    offset = 0.1  # Offset to jitter points within each cluster

    for i, model in enumerate(models):
        cluster_values = [accuracies[cluster][model] for cluster in cluster_indices]
        x_positions = [
            cluster + (i - len(models) / 2) * offset for cluster in cluster_indices
        ]
        ax1[0].scatter(x_positions, cluster_values, label=model)

    ax1[0].set_xlabel("Cluster")
    ax1[0].set_ylabel("Accuracy")
    ax1[0].set_title(
        f"Accuracy of each model in each cluster, num clusters = {len(cluster_indices)}, metric = {metric}"
    )
    ax1[0].set_xticks(ticks=range(len(cluster_indices)))
    ax1[0].set_xticklabels(cluster_labels, rotation=45)
    ax1[0].legend()
    ax1[0].grid(True)

    # Plot the cluster sizes
    cluster_sizes_values = [cluster_sizes[cluster] for cluster in cluster_indices]
    ax1[1].bar(cluster_labels, cluster_sizes_values, color="skyblue")
    ax1[1].set_xlabel("Cluster")
    ax1[1].set_ylabel("Size")
    ax1[1].set_title("Size of each cluster")
    ax1[1].set_xticklabels(cluster_labels, rotation=45)
    ax1[1].grid(True)

    plt.tight_layout()
    plt.savefig(
        f"per_cluster_accuracy_and_size_num_clusters_{len(cluster_indices)}_metric_{metric}.png"
    )
    plt.close()


def generate_result(n_clusters: int, metric: str):
    # Load vectors and compute the optimized distance matrix
    vectors = load_vectors()
    print("vectors loaded")
    dist_matrix = multiclass_distance_matrix(vectors, metric=metric)
    print("distance matrix loaded")

    # Cluster the data using K-Medoids
    cluster_labels, kmedoids = cluster_with_kmedoids(dist_matrix, n_clusters=n_clusters)
    print("Clustering completed")
    print(f"Number of examples in each cluster: {np.bincount(cluster_labels)}")

    # Save the clustered dataframe
    save_clustered_dataframe(
        vectors=vectors,
        cluster_labels=cluster_labels,
    )

    # Plot the accuracy of each model in each cluster
    plot_per_cluster_accuracy(metric=metric)


def main():
    for metric in ["chebyshev", "hamming"]:
        for n_clusters in range(2, 21):
            generate_result(n_clusters, metric)


if __name__ == "__main__":
    main()
