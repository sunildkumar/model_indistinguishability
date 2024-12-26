import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
