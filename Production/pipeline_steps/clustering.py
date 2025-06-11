import pandas as pd
from sklearn.cluster import KMeans

def apply_clustering(pcaDF: pd.DataFrame, num_components, n_clusters=3):
    kmeans = KMeans(n_clusters, random_state=42)
    for i in range(2, num_components + 1):
        runPcaDF = pcaDF.values[:, :i]
        clusters = kmeans.fit_predict(runPcaDF)
        pcaDF[f'Cluster_{i}_PCs'] = clusters
    return pcaDF
