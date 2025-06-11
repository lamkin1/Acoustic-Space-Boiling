import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
import time

# --- Copied from functions.py for self-containment ---
def unpack_data(data):
    time = data.index
    signal = data.iloc[:, 0]
    return time, signal

def get_peaks(acceleration0):
    import numpy as np
    from scipy import signal
    percentile = np.percentile(acceleration0, 99.5)
    percent_of_max = 0.1 * np.max(acceleration0)
    def clamp(value, lower=0.015, upper=0.1):
        return max(lower, min(value, upper))
    height = clamp(max(percentile, percent_of_max))
    distance = 350 + 5 / height
    x, y = signal.find_peaks(acceleration0, distance=distance, height=height)
    x = x / 10000
    y = y["peak_heights"]
    return x, y

class Candidates:
    def __init__(self, x, y, run_length, confidence=0.99, verbose=False):
        self.candidate_lst = []
        self.x_sd = 0.03
        self.y_sd = 0.04
        self.x = x
        self.y = y
        self.run_length = run_length
        self.x_margin = self.x_sd * 2.3263  # z for 99%
        self.y_margin = self.y_sd * 2.3263
        self.cur_id = 0
        self.num_peaks = len(x)
        self.p_null = min(1.0, (2 * self.x_margin / run_length) * self.num_peaks)
        self.alpha = 0.05
        self.used_peaks = set()
        self.verbose = verbose
    def add_candidate(self, candidate):
        self.candidate_lst.append(candidate)
        self.cur_id += 1
    def generate_candidates(self, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        for i in range(min(int(self.num_peaks / 2), self.num_peaks - 1)):
            for j in range(i + 1, self.num_peaks):
                d = round(self.x[j] - self.x[i], 4)
                if self.x[j] + (int(self.num_peaks ** 0.5)) * d / 2 > self.run_length + self.x_margin:
                    break
                if d < 2 * self.x_margin or self.y[i] - self.y[j] > 2 * self.y_margin:
                    continue
                anchor = self.x[i]
                candidate = Candidate(self.cur_id, d, anchor)
                self.add_candidate(candidate)
    def add_hit_data(self, verbose=None):
        for candidate in self.candidate_lst:
            candidate.count_hits(self.x, self.y, self.x_margin, self.y_margin)
    def prune_insufficient_hits(self, verbose=None):
        self.candidate_lst = [c for c in self.candidate_lst if c.hits >= 3 and c.binomial_test(self.p_null, alpha=self.alpha / max(1, len(self.candidate_lst)))]
    def group_candidates_by_similarity(self, threshold=0.75, verbose=None):
        import networkx as nx
        candidates = self.candidate_lst
        n = len(candidates)
        ids = [c.id for c in candidates]
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    sim_matrix[i][j] = 1.0
                    continue
                s1 = set(candidates[i].hit_indices or [])
                s2 = set(candidates[j].hit_indices or [])
                sim_matrix[i][j] = len(s1 & s2) / min(len(s1), len(s2)) if s1 and s2 else 0.0
        G = nx.Graph()
        G.add_nodes_from(ids)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= threshold:
                    G.add_edge(ids[i], ids[j])
        id_to_candidate = {c.id: c for c in candidates}
        dominant_candidates = []
        for group in nx.connected_components(G):
            group_cands = [id_to_candidate[i] for i in group]
            dominant = max(group_cands, key=lambda c: (len(c.hit_indices), c.hits))
            dominant.absorbed = []
            for c in group_cands:
                if c is not dominant:
                    dominant.absorbed.extend(set(c.hit_indices or []) - set(dominant.hit_indices or []))
            dominant_candidates.append(dominant)
        self.candidate_lst = dominant_candidates
    def remove_outliers(self, verbose=None):
        min_hits = max(1, int(self.num_peaks ** 0.5))
        self.candidate_lst = [c for c in self.candidate_lst if len(c.hit_indices) + len(getattr(c, 'absorbed', [])) >= min_hits]
    def count_unused_peaks(self):
        used_peaks = set()
        for candidate in self.candidate_lst:
            for peak in candidate.hit_indices:
                used_peaks.add(peak)
            for peak in getattr(candidate, 'absorbed', []):
                used_peaks.add(peak)
        return self.num_peaks - len(used_peaks)
    def get_unused_peak_proportion(self):
        return self.count_unused_peaks() / self.num_peaks
    def get_num_regimes(self):
        return len(self.candidate_lst)
    def detect_regimes(self, verbose=None):
        self.generate_candidates()
        self.add_hit_data()
        self.prune_insufficient_hits()
        self.group_candidates_by_similarity()
        self.remove_outliers()
        return self.get_num_regimes()

class Candidate:
    def __init__(self, id, d, anchor):
        self.id = id
        self.d = d
        self.anchor = anchor
        self.hit_indices = None
        self.hits = None
        self.tries = None
        self.absorbed = []
    def count_hits(self, x, y, x_margin, y_margin):
        x = np.array(x)
        y = np.array(y)
        anchor_idx = np.where(x == self.anchor)[0][0]
        hit_indices = [int(anchor_idx)]
        tries = 1
        last_hit = self.anchor
        while True:
            t = last_hit + self.d
            if t > x[-1] + x_margin:
                break
            x_mask = (x >= t - x_margin) & (x <= t + x_margin)
            candidate_idxs = np.where(x_mask)[0]
            if candidate_idxs.size > 0:
                recent_hits_y = y[hit_indices[-3:]]
                expected_y = np.mean(recent_hits_y)
                y_diff = np.abs(y[candidate_idxs] - expected_y)
                valid_mask = y_diff <= y_margin
                valid_idxs = candidate_idxs[valid_mask]
                if valid_idxs.size > 0:
                    distances = np.abs(x[valid_idxs] - t)
                    hit_idx = int(valid_idxs[np.argmin(distances)])
                    hit_indices.append(hit_idx)
                    last_hit = x[hit_idx]
                else:
                    last_hit = t
            else:
                last_hit = t
            tries += 1
        self.hit_indices = hit_indices
        self.hits = len(hit_indices)
        self.tries = tries
    def binomial_test(self, p_null, alpha=0.01):
        if not self.hits:
            raise RuntimeError("Must call `count_hits()` before `binomial_test()`")
        if self.hits < 3:
            return False
        from scipy.stats import binom
        p_value = 1 - binom.cdf(self.hits - 2, self.tries - 2, p_null)
        return p_value < alpha

def get_boilings_data(x, y, run_length, verbose=False):
    if len(x) < 3:
        return 0, 0
    if len(x) > 300:
        return 3, 0
    candidates = Candidates(x, y, run_length)
    candidates.detect_regimes(verbose=verbose)
    num_boilings = candidates.get_num_regimes()
    unused_peak_proportion = candidates.get_unused_peak_proportion()
    return min(num_boilings, 3), unused_peak_proportion
# --- End of copied code ---

def compute_spectral_entropy(signal_data, fs=1000):
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    psd_norm = psd / np.sum(psd)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return entropy

def compute_spectral_centroid(signal_data, fs=1000):
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    centroid = np.sum(freqs * psd) / np.sum(psd)
    return centroid

def compute_spectral_flatness(signal_data, fs=1000):
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    geometric_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    arithmetic_mean = np.mean(psd)
    flatness = geometric_mean / arithmetic_mean
    return flatness

def compute_spectral_bandwidth(signal_data, fs=1000):
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    centroid = np.sum(freqs * psd) / np.sum(psd)
    bandwidth = np.sqrt(np.sum(psd * (freqs - centroid) ** 2) / np.sum(psd))
    return bandwidth

def extract_peak_features(file):
    data = pd.read_csv(file, index_col="Time")
    time, signal_data = unpack_data(data)
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
    run_length = len(time) / 10000
    num_boilings, unused_peak_proportion = get_boilings_data(peaks, magnitude, run_length)
    project_root = Path(__file__).resolve().parent.parent
    file_path = Path(file).resolve()
    try:
        rel_file = file_path.relative_to(project_root)
        file_name_str = str(rel_file)
    except ValueError:
        file_name_str = file_path.name
    return {
        "file_name": file_name_str,
        "std_dev_time_diff": std_dev_time_diff,
        "mean_time_diff": mean_time_diff,
        "median_time_diff": median_time_diff,
        "max_peak": max_peak,
        "median_peak": median_peak,
        "std_peak": std_peak,
        "avg_peaks_per_second": avg_peaks_per_second,
        "sum_peak_magnitude": sum_peak_magnitude,
        "percent_time_above_threshold": percent_time_above_threshold,
        "num_boilings": num_boilings,
        "unused_peak_proportion": unused_peak_proportion
    }

def extract_all_features(file):
    data = pd.read_csv(file, index_col="Time")
    _, signal_data = unpack_data(data)
    # Use the file path as a string relative to the workspace root if possible, else just the filename
    project_root = Path(__file__).resolve().parent.parent
    file_path = Path(file).resolve()
    try:
        rel_file = file_path.relative_to(project_root)
        file_name_str = str(rel_file)
    except ValueError:
        file_name_str = file_path.name
    features = {
        "file_name": file_name_str,
        "spectral_entropy": compute_spectral_entropy(signal_data),
        "spectral_centroid": compute_spectral_centroid(signal_data),
        "spectral_flatness": compute_spectral_flatness(signal_data),
        "spectral_bandwidth": compute_spectral_bandwidth(signal_data)
    }
    peak_features = extract_peak_features(file)
    if peak_features:
        features.update(peak_features)
    return features

def process_directory(directory_name="Production/labeled_runs/", verbose=False):
    script_dir = Path(__file__).resolve().parent
    directory = (script_dir / ".." / directory_name).resolve()
    extracted_features = []
    for f in directory.iterdir():
        if f.suffix == '.csv':
            start = time.time()
            extracted_features.append(extract_all_features(f))
            if verbose:
                print(f"Extracted features from {f} in {round(time.time() - start, 2)} seconds.")
    feature_df = pd.DataFrame(extracted_features)
    feature_df.fillna(0, inplace=True)
    feature_df.to_csv("features.csv", index=False)
    print("Features saved successfully to 'features.csv'!")
