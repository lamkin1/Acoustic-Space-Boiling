from pipeline_steps.file_management import move_new_runs_by_labels
from pipeline_steps.scaling import preprocess
from pipeline_steps.pca import apply_pca
from pipeline_steps.clustering import apply_clustering
from pipeline_steps.date_utils import create_date_from_string
from pipeline_steps.decision_tree import retrain_decision_tree
from pipeline_steps.feature_extraction import extract_all_features

import pandas as pd
import os

def main():
    # Always sort new_runs into labeled_runs and unlabeled_runs
    move_new_runs_by_labels()

    # Ask user for all inputs at the beginning
    print("Retraining unsupervised learning (feature extraction + clustering) on all files in labeled_runs and unlabeled_runs.")
    retrain_tree = input("Do you want to retrain the supervised decision tree on labeled runs? (y/n): ").strip().lower()

    # Step 1: Feature extraction for all runs
    data_dirs = ["labeled_runs", "unlabeled_runs"]
    import glob
    from pathlib import Path
    all_files = []
    for d in data_dirs:
        dir_path = Path(d)
        all_files.extend([str(f) for f in dir_path.glob('*.csv')])
    
    import time
    extracted_features = []
    for f in all_files:
        start = time.time()
        features = extract_all_features(Path(f))
        # Store only the child file name (no parent folders)
        features["file_name"] = Path(features["file_name"]).name
        extracted_features.append(features)
    feature_df = pd.DataFrame(extracted_features)
    feature_df.fillna(0, inplace=True)
    features_csv_path = os.path.join("app_resources", "features.csv")
    feature_df.to_csv(features_csv_path, index=False)
    print(f"Features saved successfully to '{features_csv_path}'!")

    # Step 2: Scaling, PCA, Clustering, and Output
    output_dir = "app_resources/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = pd.read_csv(features_csv_path)
    file_names = data["file_name"].copy()
    scaled_data, remaining_column_names = preprocess(data)
    pca_data, loadings_matrix = apply_pca(scaled_data, n_components=6, previous_column_names=remaining_column_names, verbose=True)
    for num_clusters in range(2, 20):
        clustered_df = apply_clustering(pca_data, 6, n_clusters=num_clusters)
        clustered_df["file_name"] = file_names
        clustered_df['date'] = clustered_df['file_name'].apply(create_date_from_string)
        clustered_df.sort_values(by="date", inplace=True)
        output_path_df = os.path.join(output_dir, f'results_{num_clusters}_clusters.csv')
        output_path_loadings = os.path.join(output_dir, f'loadings.csv')
        clustered_df.to_csv(output_path_df, index=False)
        if loadings_matrix.index.name != 'Feature':
            loadings_matrix.index.name = 'Feature'
        loadings_matrix.to_csv(output_path_loadings)
        print(f"Scaled data saved to: {output_path_df} & {output_path_loadings}")

    # Supervised learning (decision tree) only if user requested
    if retrain_tree == "y":
        print("Retraining supervised decision tree on labeled_runs only.")
        # Cross-reference features.csv and labels.csv to only use labeled runs
        features_df = pd.read_csv(features_csv_path)
        # Use absolute path for labels.csv in app_resources
        labels_csv = os.path.join("app_resources", "labels.csv")
        if not os.path.exists(labels_csv):
            print(f"Error: labels.csv not found at {labels_csv}. Skipping supervised retraining.")
            return
        labels_df = pd.read_csv(labels_csv)
        features_df['file_name'] = features_df['file_name'].str.strip().str.lower()
        labels_df['file_name'] = labels_df['file_name'].str.strip().str.lower()
        merged_features = pd.merge(features_df, labels_df, on="file_name", how="inner")
        # merged_features_csv = os.path.join("app_resources/output", "labeled_features.csv")
        # merged.to_csv(merged_features_csv, index=False)
        model_out = os.path.join(output_dir, "tree_model.pkl")
        tree_png = os.path.join(output_dir, "tree_plot.png")
        # Only pass merged_features_csv to retrain_decision_tree
        retrain_decision_tree(merged_features, model_out, tree_png)
    else:
        print("Supervised decision tree retraining skipped.")

if __name__ == "__main__":
    main()