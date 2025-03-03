import pandas as pd
import numpy as np
import sys
import os
import getopt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def preprocess(data):
    """
    Scales the data using StandardScaler, removing string and NA columns.
    Args:
        data (pd.DataFrame): The input data to be scaled.
    Returns:
        np.ndarray: The scaled data.
    """
    string_cols = []
    na_cols = []
    numeric_cols = []

    for col in data.columns:
        if pd.api.types.is_string_dtype(data[col]):
            string_cols.append(col)
        elif data[col].isnull().any():
            na_cols.append(col)
        else:
            numeric_cols.append(col)

    removed_cols = string_cols + na_cols

    if removed_cols:
        print("Removed columns:", removed_cols)
        data = data[numeric_cols]  # Keep only numeric and non-NA columns

    scaler = StandardScaler()
    scaledData = scaler.fit_transform(data)
    return scaledData, numeric_cols

def apply_pca(scaledData, n_components, previous_column_names, verbose = False):
    # Apply PCA
    pca = PCA(n_components = n_components)  # Reduce to 2 components for 2D visualization
    pcaData = pca.fit_transform(scaledData)

    # Create column names based on the number of components
    column_names = [f'PCA{i+1}' for i in range(n_components)]

    pcaDF = pd.DataFrame(data = pcaData, columns = column_names)

    # Create loading matrix 
    loading_matrix = pd.DataFrame(
        pca.components_.T,
        columns=column_names,
        index=previous_column_names
    )

    if(verbose):
        variance_ratios = pca.explained_variance_ratio_
        print("\nVariance explained by each principal component:\n")
        for i, ratio in enumerate(variance_ratios, start=1):
            print(f"PC{i}: {ratio:.2%}")
        print('')
        print('Loadings Matrix:\n',loading_matrix)
        print('')

    return pcaDF, loading_matrix

def apply_clustering(pcaDF: pd.DataFrame, n_clusters = 3):
    kmeans = KMeans(n_clusters, random_state = 42)
    clusters = kmeans.fit_predict(pcaDF.values)

    # Add cluster prediction to each observation
    pcaDF['Cluster'] = clusters

    return pcaDF
    

def main():
    """
    Extracts command-line arguments, reads a CSV file, performs standard scaling,
    and handles file/directory errors.
    """

    args = sys.argv[1:]

    # Define the expected number of arguments
    expected_args = 2


    try:
        opts, args = getopt.getopt(sys.argv[1:], "v")
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    verbose = False
    for opt, arg in opts:
        if opt == "-v":
            verbose = True

    if verbose:
        print("Verbose mode enabled.")

    if len(args) != expected_args:
        print(f"Error: Expected {expected_args} arguments, but got {len(args)}.")
        print(f"Usage: python {sys.argv[0]} features.csv output_directory")
        sys.exit(1)

    input_csv = args[0]
    output_dir = args[1]

    # Check if the input file exists
    if not os.path.isfile(input_csv):
        print(f"Error: File '{input_csv}' not found.")
        sys.exit(1)

    # Check if the output directory exists
    if not os.path.isdir(output_dir):
        print(f"Error: Directory '{output_dir}' not found.")
        sys.exit(1)

    try:
        data = pd.read_csv(input_csv)
    except Exception as e: #more general error handling
        print(f"Error: Could not read CSV file '{input_csv}'. {e}")
        sys.exit(1)

    
    try:
        file_names = data["file_name"].copy()

        # Standard Scaler
        scaled_data, remaining_column_names = preprocess(data)

        print("Completed Scaling")

        # Apply PCA and retrieve loadings matrix
        pca_data, loadings_matrix = apply_pca(scaled_data, n_components = 6,  previous_column_names = remaining_column_names, verbose=verbose)

        print("Completed PCA")

        # Apply Clustering (KMeans as of now)
        clustered_df = apply_clustering(pca_data, n_clusters = 3)

        print("Completed Clustering")

        # Add Filename to final clustered dataframe
        clustered_df["file_name"] = file_names

        # Convert to csv file
        output_path_df, output_path_loadings = os.path.join(output_dir, "clusteredDF.csv"), os.path.join(output_dir, "loadings.csv")
        clustered_df.to_csv(output_path_df, index=False) #save it to the output directory

        loadings_matrix.reset_index(inplace=True)
        loadings_matrix.rename(columns={'index': 'Feature'}, inplace=True)
        loadings_matrix.to_csv(output_path_loadings, index=False)
        
        print(f"Scaled data saved to: {output_path_df} & {output_path_loadings}")

    except Exception as e: #more general error handling.
        print(f"Error: Runetime error. {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()