import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from cluster import (
    cluster_with_kmedoids,
    load_vectors,
    multiclass_distance_matrix,
    save_clustered_dataframe,
)

# investigates the interesting cluster when clustering with chebyshev distance and 10 clusters
vectors = load_vectors()
dist_matrix = multiclass_distance_matrix(vectors, metric="chebyshev")
cluster_labels, kmedoids = cluster_with_kmedoids(dist_matrix, n_clusters=10)
save_clustered_dataframe(vectors=vectors, cluster_labels=cluster_labels)

# load the clustered dataframe
clustered_df = pd.read_csv("model_eval_results_with_clusters.csv")


# cluster 0 is the interesting cluster
cluster_0_df = clustered_df[clustered_df["Cluster Label"] == 0]

# Define the model columns
model_columns = [
    "ViT Label",
    "ConvNext Label",
    "Swin Label",
    "ViT16Lora Label",
    "ResNet Label",
]

# Initialize a dictionary to store misclassification counts
mistakes = {
    model: {true_class: 0 for true_class in cluster_0_df["True label"].unique()}
    for model in model_columns
}

# Calculate misclassifications for each model
for model in model_columns:
    y_true = cluster_0_df["True label"]
    y_pred = cluster_0_df[model]

    # Update misclassification counts
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            mistakes[model][true_label] += 1

# Calculate the proportion of mistakes by model by class
proportions = {}
for model, class_mistakes in mistakes.items():
    total_mistakes = sum(class_mistakes.values())
    proportions[model] = {
        true_class: count / total_mistakes if total_mistakes > 0 else 0
        for true_class, count in class_mistakes.items()
    }


class_names = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}

# Convert the proportions dictionary to a DataFrame for easier plotting
proportions_df = pd.DataFrame(proportions)

# Update the index of the DataFrame to use class names
proportions_df.index = proportions_df.index.map(class_names)

# Clean up the column names by removing " Label"
proportions_df.columns = [col.replace(" Label", "") for col in proportions_df.columns]

# Plot heatmap showing the proportion of mistakes by model by class
plt.figure(figsize=(10, 8))
sns.heatmap(
    proportions_df,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    cbar_kws={"label": "Proportion of Mistakes"},
)
plt.title("Proportion of Mistakes by True Class and Model")
plt.xlabel("Model")
plt.ylabel("True Class")
if not os.path.exists("other_plots"):
    os.makedirs("other_plots", exist_ok=True)
plt.savefig("other_plots/proportion_mistakes_heatmap.png")
plt.close()

# Calculate accuracy for each example
cluster_0_df["Correct Predictions"] = cluster_0_df.apply(
    lambda row: sum(row[model] == row["True label"] for model in model_columns), axis=1
)

cluster_0_df["Accuracy"] = cluster_0_df["Correct Predictions"] / len(model_columns)

# Sort by accuracy to see the most challenging examples
sorted_results = cluster_0_df.sort_values("Accuracy")[
    ["filename", "True label", "Accuracy"] + model_columns
]

print("\nAccuracy Statistics:")
print(f"Mean accuracy: {cluster_0_df['Accuracy'].mean():.3f}")
print(f"Median accuracy: {cluster_0_df['Accuracy'].median():.3f}")
print("\nMost challenging examples (lowest accuracy):")
print(sorted_results.head())

# select those with 0.0 accuracy
zero_acc_rows = sorted_results[(sorted_results["Accuracy"] == 0.0)]

# make a grid of images where no model predicts the correct class
fnames = zero_acc_rows["filename"].tolist()
true_labels = zero_acc_rows["True label"].tolist()
true_label_names = zero_acc_rows["True label"].map(class_names).tolist()

# Sort images by true label
sorted_indices = sorted(range(len(true_labels)), key=lambda k: true_labels[k])
fnames = [fnames[i] for i in sorted_indices]
true_label_names = [true_label_names[i] for i in sorted_indices]
images = [Image.open(f"data/cifar-10-test/{fname}") for fname in fnames]

# Calculate grid dimensions
n_images = len(images)
grid_size = int(np.ceil(np.sqrt(n_images)))

# Create subplot grid
fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
fig.suptitle("Images with 0% Accuracy Across All Models", fontsize=16)

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Plot each image
for idx, (img, label) in enumerate(zip(images, true_label_names)):
    ax = axes_flat[idx]
    ax.imshow(img)
    ax.set_title(label, fontsize=10, pad=8, y=-0.15)  # Move title below image
    ax.axis("off")

# Hide empty subplots
for idx in range(len(images), len(axes_flat)):
    axes_flat[idx].axis("off")

plt.tight_layout()
plt.savefig("other_plots/challenging_examples_grid.png", dpi=300, bbox_inches="tight")
plt.close()
