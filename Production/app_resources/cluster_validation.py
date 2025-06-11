import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import adjusted_rand_score

def similarity_matrix(true_labels, cluster_labels, metric="jaccard"):
    """
    Computes a similarity matrix between true class labels and cluster labels
    using the specified metric: 'jaccard' or 'ss' (Szymkiewicz–Simpson).

    Parameters:
        true_labels (array-like): Ground truth class labels.
        cluster_labels (array-like): Cluster labels (e.g., from KMeans).
        metric (str): Similarity metric to use, 'jaccard' or 'ss'.

    Returns:
        similarity_matrix (ndarray): Matrix of shape (n_classes, n_clusters),
                                     where [i,j] is the similarity between class i and cluster j.
    """
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    unique_classes = np.unique(true_labels)
    unique_clusters = np.unique(cluster_labels)

    sim_matrix = np.zeros((len(unique_classes), len(unique_clusters)))

    for i, c in enumerate(unique_classes):
        for j, k in enumerate(unique_clusters):
            y_true = (true_labels == c)
            y_pred = (cluster_labels == k)

            intersection = np.logical_and(y_true, y_pred).sum()
            union = np.logical_or(y_true, y_pred).sum()
            min_size = min(y_true.sum(), y_pred.sum())

            if metric == "jaccard":
                sim_matrix[i, j] = intersection / union if union != 0 else 0.0
            elif metric == "ss":
                sim_matrix[i, j] = intersection / min_size if min_size != 0 else 0.0
            else:
                raise ValueError("Metric must be 'jaccard' or 'ss' (Szymkiewicz–Simpson)")

    return sim_matrix


def plot_similarity_matrix(true_labels, cluster_labels, metric="jaccard", title=None):
    """
    Computes and plots a similarity matrix between class and cluster labels.

    Parameters:
        true_labels (array-like): Ground truth class labels.
        cluster_labels (array-like): Cluster labels (e.g., from KMeans).
        metric (str): Similarity metric to use, 'jaccard' or 'ss'.
        title (str): Optional title for the plot.
    """
    sim_matrix = similarity_matrix(true_labels, cluster_labels, metric=metric)

    # Label rows and columns
    class_labels = [f"Class {i}" for i in np.unique(true_labels)]
    cluster_labels_display = [f"Cluster {j}" for j in np.unique(cluster_labels)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=cluster_labels_display, yticklabels=class_labels)
    plt.xlabel("Clusters")
    plt.ylabel("True Classes")
    plt.title(title or f"{metric.capitalize()} Similarity Matrix")
    plt.tight_layout()
    plt.show()




def extract_clean_filename(path_str):
    """
    Extracts the meaningful filename by removing path, 'MATLAB' prefix, and trailing image extensions like '.png'.
    Keeps the original '.csv' suffix if present.

    Examples:
        'Data/After_May/MATLAB 1-00 PM Fri, Jun 28, 2024 Run8 .csv' -> '1-00 PM Fri, Jun 28, 2024 Run8 .csv'
        '1-00 PM Fri, Jun 28, 2024 Run8 .csv' -> '1-00 PM Fri, Jun 28, 2024 Run8 .csv'
        'MATLAB 1-00 PM Fri, Jun 28, 2024 Run8 .csv.png' -> '1-00 PM Fri, Jun 28, 2024 Run8 .csv'
    """
    filename = os.path.basename(path_str)

    # Remove 'MATLAB' prefix if present
    if filename.startswith("MATLAB "):
        filename = filename[len("MATLAB "):]

    # Remove trailing .png or similar extensions if the true name ends in .csv
    match = re.search(r"(.*\.csv)\b", filename)
    if match:
        return match.group(1)

    # If no .csv match, return cleaned filename anyway
    return filename


k_clusters = 7
n_pcs = 3
df_labels = pd.read_csv("parsed_labels.csv")
df_labels["filename"] = df_labels["filename"].map(extract_clean_filename)
df_clusters = pd.read_csv(f"../output/results_{str(k_clusters)}_clusters.csv")
df_clusters["file_name"] = df_clusters["file_name"].map(extract_clean_filename)

full = pd.merge(df_labels, df_clusters, left_on='filename', right_on='file_name', how='inner')
print(len(df_labels), len(df_clusters), len(full))

true_labels = full["label"].values
cluster_labels = full[f"Cluster_{n_pcs}_PCs"].values

plot_similarity_matrix(true_labels, cluster_labels, metric="ss")

print(adjusted_rand_score(true_labels, cluster_labels))