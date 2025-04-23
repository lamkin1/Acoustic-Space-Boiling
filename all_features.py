import numpy as np
import pandas as pd
import scipy.signal as signal
import pywt
import hurst
from pathlib import Path
from scipy.stats import binomtest
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from sklearn.cluster import DBSCAN
from math import sqrt
from scipy.stats import norm


# Feature Extraction Functions
def unpack_data(data):
    """
    Unpacks dataframe into time and signal
    """
    time = data.index
    signal = data.iloc[:, 0]
    return time, signal


def compute_spectral_entropy(signal_data, fs=1000):
    """
    Compute spectral entropy from the power spectral density (PSD).
    """
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    psd_norm = psd / np.sum(psd)  # Normalize power spectrum
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))  # Compute entropy
    return entropy


def compute_spectral_centroid(signal_data, fs=1000):
    """
    Computes the spectral centroid, indicating the center of mass of the frequency spectrum.
    """
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    centroid = np.sum(freqs * psd) / np.sum(psd)
    return centroid


def compute_spectral_flatness(signal_data, fs=1000):
    """
    Computes spectral flatness, which measures the noisiness of the signal in the frequency domain.
    """
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    geometric_mean = np.exp(np.mean(np.log(psd + 1e-12)))  # Avoid log(0)
    arithmetic_mean = np.mean(psd)
    flatness = geometric_mean / arithmetic_mean
    return flatness


def compute_spectral_bandwidth(signal_data, fs=1000):
    """
    Computes spectral bandwidth, which quantifies the spread of the frequency spectrum.
    """
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    centroid = np.sum(freqs * psd) / np.sum(psd)
    bandwidth = np.sqrt(np.sum(psd * (freqs - centroid) ** 2) / np.sum(psd))
    return bandwidth


def clamp(value, lower=0.015, upper=0.1):
    return max(lower, min(value, upper))


def get_peaks(acceleration0):
    """
    acceleration0: accelerometer data, measured to the 10,000th of a second
    Returns: x and y coordinates of peaks
    """
    percentile = np.percentile(acceleration0, 99.5)
    percent_of_max = 0.1*np.max(acceleration0)

    height = clamp(max(percentile, percent_of_max))
    distance = 350 + 5 / height

    x, y = signal.find_peaks(acceleration0, distance=distance, height=height)
    x = x / 10000
    y = y["peak_heights"]  # unpack data
    return x, y


def extract_peak_features(file):
    """
    Extracts peak-related features from the signal.
    """
    data = pd.read_csv(file, index_col="Time")
    _, signal_data = unpack_data(data)

    peaks, magnitude = get_peaks(signal_data)

    if len(peaks) <= 2:
        return None

    magnitude = np.array(signal_data.loc[peaks])

    time_differences = np.diff(peaks) if len(peaks) > 1 else [np.nan]
    std_dev_time_diff = np.std(time_differences)
    mean_time_diff = np.mean(time_differences)
    median_time_diff = np.median(time_differences)

    max_peak = np.max(magnitude) if len(magnitude) > 0 else np.nan
    median_peak = np.median(magnitude) if len(magnitude) > 0 else np.nan
    std_peak = np.std(magnitude) if len(magnitude) > 0 else np.nan
    avg_peaks_per_second = np.mean(peaks) if len(peaks) > 0 else np.nan
    sum_peak_magnitude = np.sum(magnitude) if len(magnitude) > 0 else np.nan

    percent_time_above_threshold = np.mean(signal_data > min(magnitude))

    return {
        "file_name": file,
        "std_dev_time_diff": std_dev_time_diff,
        "mean_time_diff": mean_time_diff,
        "median_time_diff": median_time_diff,
        "max_peak": max_peak,
        "median_peak": median_peak,
        "std_peak": std_peak,
        "avg_peaks_per_second": avg_peaks_per_second,
        "sum_peak_magnitude": sum_peak_magnitude,
        "percent_time_above_threshold": percent_time_above_threshold,
        # "mean_post_peak_magnitude": mean_post_peak_magnitude,
        # "std_dev_post_peak_magnitude": std_dev_post_peak_magnitude
    }


def extract_all_features(file):
    """
    Extracts both peak-related and spectral features from the given file.
    """
    data = pd.read_csv(file, index_col="Time")
    _, signal_data = unpack_data(data)

    # Extract spectral and statistical features
    features = {
        "file_name": file,
        "spectral_entropy": compute_spectral_entropy(signal_data),
        "spectral_centroid": compute_spectral_centroid(signal_data),
        "spectral_flatness": compute_spectral_flatness(signal_data),
        "spectral_bandwidth": compute_spectral_bandwidth(signal_data)
    }

    # Extract peak-based features
    peak_features = extract_peak_features(file)
    if peak_features:
        features.update(peak_features)

        # Calculate number of boilings using the peaks
        peaks, _ = get_peaks(signal_data)
        if len(peaks) > 2:  # Only calculate if we have enough peaks
            features["number_of_boilings"] = number_of_boilings(peaks)
        else:
            features["number_of_boilings"] = 0

    return features


def process_directory(directory_name="Data/After_May/"):
    """
    Processes all CSV files in a directory and extracts all features from each.
    """
    directory = Path(directory_name)
    file_names = [f for f in directory.iterdir() if f.suffix == '.csv']

    extracted_features = [extract_all_features(file) for file in file_names]

    # Convert to DataFrame
    feature_df = pd.DataFrame(extracted_features)
    #feature_df.dropna(inplace=True)  # remove runs with 2 or fewer peaks
    feature_df.fillna(0, inplace=True) # replace all na with 0s
    feature_df.to_csv("features.csv", index=False)
    print(f"Features saved successfully to 'features.csv'!")


def get_diffs(x):
    min_tries = 3
    max_allowed_diff = (x[-1] - x[0]) / 2
    max_diff = (x[-1] - x[0]) / (min_tries - 1)  # e.g., 3 tries = 2 intervals

    # Use the tighter of the two constraints
    cutoff = min(max_diff, max_allowed_diff)

    diffs = []
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            d = float(x[j] - x[i])
            if d > cutoff:
                break  # since x is sorted, all future j will be worse
            diffs.append(d)

    return sorted(diffs)


def pmm_clustering(diffs, random_state=0):
    """
    Performs GMM clustering on diffs and returns a dictionary of clusters sorted by mean.
    Each cluster is represented as a dictionary with keys:
    - 'mean': mean of the cluster
    - 'sd': standard deviation of the cluster
    - 'n': number of points in the cluster
    """
    ## EVALUATE
    if len(diffs) < 2:
        return {}  # Return empty dict if not enough data points

    min_components = 1
    max_components = min(50, max(2, len(diffs) // 5))

    X = np.array(diffs).reshape(-1, 1)
    bics, models = [], []

    for k in range(min_components, max_components + 1):
        model = GaussianMixture(n_components=k, random_state=random_state).fit(X)
        bics.append(model.bic(X))
        models.append(model)

    # Select best model
    best_index = np.argmin(bics)
    best_model = models[best_index]
    labels = best_model.predict(X)

    # Group values by cluster label
    cluster_data = defaultdict(list)
    for val, label in zip(diffs, labels):
        cluster_data[label].append(val)

    # Summarize and sort clusters by mean
    raw_clusters = []
    for values in cluster_data.values():
        values = np.array(values)
        if len(values) < 2:
            continue
        raw_clusters.append({
            'mean': np.mean(values),
            'sd': np.std(values),
            'n': len(values)
        })

    raw_clusters.sort(key=lambda x: x['mean'])

    # Build final result
    clusters = {
        i: {
            'mean': np.float64(c['mean']),
            'sd': np.float64(c['sd']),
            'n': int(c['n'])
        } for i, c in enumerate(raw_clusters)
    }

    return clusters


def count_hits_with_optimal_start(x, delta_t, sd, n, confidence=0.95, max_starts=10, extra_margin=0.00405):
    x = np.array(x)
    se = sd / np.sqrt(n)
    z = norm.ppf(1 - (1 - confidence) / 2)
    margin = z * se + extra_margin

    best_hits = []
    best_tries = 0
    best_anchor = x[0]

    for i in range(min(max_starts, len(x))):
        anchor = x[i]
        hit_indices = []
        tries = 0
        last_hit = anchor

        while True:
            t = last_hit + delta_t
            if t > x[-1]:
                break

            # Check for hit within margin
            candidates = np.where((x >= t - margin) & (x <= t + margin))[0]
            if candidates.size > 0:
                hit_idx = candidates[0]
                hit_indices.append(hit_idx)
                last_hit = x[hit_idx]  # Recenter from hit
            else:
                last_hit = t  # No hit — continue from expected

            tries += 1

        if len(hit_indices) > len(best_hits):
            best_hits = hit_indices
            best_tries = tries
            best_anchor = anchor

    return delta_t, best_anchor, len(best_hits), best_tries


def add_hit_data(clusters, x):
    for id, cluster in clusters.items():
        delta_t, anchor, hits, tries = count_hits_with_optimal_start(x, cluster["mean"], cluster["sd"], cluster["n"])
        cluster["anchor"] = anchor
        cluster["hits"] = hits
        cluster["tries"] = tries
    return clusters


def filter_delta_ts(hit_data, alpha=0.05, p_null=0.3):
    return {
        k: v for k, v in hit_data.items()
        ## EVALUATE
        if v["tries"] >= 1 and binomtest(v["hits"], v["tries"], p_null, alternative="greater").pvalue < alpha
    }


def prob_integer_multiple(base, sd_base, n_base, candidate, sd_candidate, n_candidate):
    ratio = candidate / base
    best_k = round(ratio)

    # Use standard error (not just std dev)
    se_base = sd_base / sqrt(n_base)
    se_candidate = sd_candidate / sqrt(n_candidate)

    # Error propagation for ratio
    ratio_se = ratio * sqrt((se_candidate / candidate)**2 + (se_base / base)**2)

    z = abs(ratio - best_k) / ratio_se
    probability = 2 * (1 - norm.cdf(z))  # two-tailed

    return probability


def remove_integer_multiples(clusters_dict, p_thresh=0.05):
    cluster_ids = list(clusters_dict.keys())
    keep = {cid: True for cid in cluster_ids}

    for i in range(len(cluster_ids) - 1):
        ci_id = cluster_ids[i]
        if not keep[ci_id]:
            continue
        for j in range(i + 1, len(cluster_ids)):
            cj_id = cluster_ids[j]
            if not keep[cj_id]:
                continue

            ci = clusters_dict[ci_id]
            cj = clusters_dict[cj_id]

            p = prob_integer_multiple(
                base=ci["mean"], sd_base=ci["sd"], n_base=ci["n"],
                candidate=cj["mean"], sd_candidate=cj["sd"], n_candidate=cj["n"]
            )

            if p > p_thresh:
                keep[cj_id] = False

    # Return a new dictionary with only the clusters to keep
    pruned = {cid: clusters_dict[cid] for cid in cluster_ids if keep[cid]}
    return pruned


def number_of_boilings(peaks):
    diffs = get_diffs(peaks)
    clusters = pmm_clustering(diffs)

    ## EVALUATE
    if not clusters:  # If no clusters found
        return 0


    clusters = add_hit_data(clusters, peaks)
    filtered_clusters = filter_delta_ts(clusters)
    base_diffs = remove_integer_multiples(filtered_clusters)
    return len(base_diffs)


if __name__ == "__main__":
    process_directory()