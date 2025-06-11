import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(scaledData, n_components, previous_column_names, verbose=False):
    pca = PCA(n_components=n_components)
    pcaData = pca.fit_transform(scaledData)
    column_names = [f'PCA{i+1}' for i in range(n_components)]
    pcaDF = pd.DataFrame(data=pcaData, columns=column_names)
    loading_matrix = pd.DataFrame(
        pca.components_.T,
        columns=column_names,
        index=previous_column_names
    )
    if verbose:
        variance_ratios = pca.explained_variance_ratio_
        print("\nVariance explained by each principal component:\n")
        for i, ratio in enumerate(variance_ratios, start=1):
            print(f"PC{i}: {ratio:.2%}")
        print('')
        print('Loadings Matrix:\n', loading_matrix)
        print('')
    return pcaDF, loading_matrix
