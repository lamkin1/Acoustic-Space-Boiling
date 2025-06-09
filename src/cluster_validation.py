import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import adjusted_rand_score

def jaccard_similarity_matrix(true_labels, cluster_labels):
    """
    Computes a Jaccard similarity matrix between true class labels and cluster labels.

    Parameters:
        true_labels (array-like): Ground truth class labels.
        cluster_labels (array-like): Cluster labels (e.g., from KMeans).

    Returns:
        jaccard_matrix (ndarray): Matrix of shape (n_classes, n_clusters), where each element [i,j]
                                   is the Jaccard similarity between class i and cluster j.
    """
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    unique_classes = np.unique(true_labels)
    unique_clusters = np.unique(cluster_labels)

    jaccard_matrix = np.zeros((len(unique_classes), len(unique_clusters)))

    for i, c in enumerate(unique_classes):
        for j, k in enumerate(unique_clusters):
            y_true_bin = (true_labels == c).astype(int)
            y_cluster_bin = (cluster_labels == k).astype(int)
            jaccard_matrix[i, j] = jaccard_score(y_true_bin, y_cluster_bin)

    return jaccard_matrix



def plot_jaccard_similarity(true_labels, cluster_labels, title="Jaccard Similarity Matrix"):
    """
    Computes and plots the Jaccard similarity matrix between class and cluster labels.

    Parameters:
        true_labels (array-like): Ground truth class labels.
        cluster_labels (array-like): Cluster labels (e.g., from KMeans).
        title (str): Title for the heatmap plot.
    """
    jaccard_matrix = jaccard_similarity_matrix(true_labels, cluster_labels)

    # Label rows and columns
    class_labels = [f"Class {i}" for i in np.unique(true_labels)]
    cluster_labels = [f"Cluster {j}" for j in np.unique(cluster_labels)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(jaccard_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=cluster_labels, yticklabels=class_labels)
    plt.xlabel("Clusters")
    plt.ylabel("True Classes")
    plt.title(title)
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

plot_jaccard_similarity(true_labels, cluster_labels)

print(adjusted_rand_score(true_labels, cluster_labels))