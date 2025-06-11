from src.utils import *
from math import sqrt
from scipy.stats import norm
from scipy.stats import binom
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def get_x_sd():
    return 0.03


def get_y_sd():
    return 0.04


def calculate_x_margin(x_sd, confidence):
    z = norm.ppf(confidence)  # one-sided
    margin = z * x_sd
    return margin


def calculate_y_margin(y_sd, confidence):
    z = norm.ppf(confidence)  # one-sided
    margin = z * y_sd
    return margin


def calculate_p_null(margin, num_peaks, run_length):
    window_fraction = (2 * margin) / run_length
    p_null = 1 - (1 - window_fraction) ** num_peaks
    return min(p_null, 1.0)


class Candidates:
    def __init__(self, x, y, run_length, confidence=0.99, verbose=False):
        self.candidate_lst = []
        self.x_sd = get_x_sd()
        self.y_sd = get_y_sd()
        self.x = x
        self.y = y
        self.run_length = run_length
        self.x_margin = calculate_x_margin(self.x_sd, confidence)
        self.y_margin = calculate_y_margin(self.y_sd, confidence)
        self.cur_id = 0
        self.num_peaks = len(x)
        self.p_null = calculate_p_null(self.x_margin, self.num_peaks, self.run_length)
        self.alpha = 0.05
        self.used_peaks = set()
        self.verbose = verbose

    def __repr__(self):
        return "\n".join(repr(c) for c in self.candidate_lst)

    def add_candidate(self, candidate):
        self.candidate_lst.append(candidate)
        self.cur_id += 1

    def generate_candidates(self, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        for i in range(self.get_max_start_index()):
            for j in range(i + 1, self.num_peaks):
                d = round(self.x[j] - self.x[i], 4)
                if self.x[j] + self.get_min_hits()*d / 2 > self.run_length + self.x_margin:
                    break
                if d < 2*self.x_margin or self.y[i] - self.y[j] > 2*self.y_margin:
                    continue
                anchor = self.x[i]
                candidate = Candidate(self.cur_id, d, anchor)
                self.add_candidate(candidate)
                if verbose:
                    print(f"Generated candidate {candidate.id}: d={d}, anchor={anchor}")
        if verbose:
            print(f"Total candidates generated: {len(self.candidate_lst)}")

    def add_hit_data(self, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        for candidate in self.candidate_lst:
            candidate.count_hits(self.x, self.y, self.x_margin, self.y_margin)
            if verbose:
                print(f"Candidate {candidate.id}: d={candidate.d}, hits={candidate.hits}, tries={candidate.tries}, indices={candidate.hit_indices}")

    def prune_insufficient_hits(self, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        original_count = len(self.candidate_lst)
        self.candidate_lst = [
            candidate for candidate in self.candidate_lst
            if candidate.hits >= 3
            and candidate.binomial_test(
                self.p_null,
                alpha=self.alpha / self.get_num_regimes() # Bonferroni Adjustment
            )
        ]
        if verbose:
            print(f"Pruned {original_count - len(self.candidate_lst)} candidates by significance")
            for candidate in self.candidate_lst:
                print(f"Candidate {candidate.id}: d={candidate.d}, hits={candidate.hits}, tries={candidate.tries}, indices={candidate.hit_indices}")

    def group_candidates_by_similarity(self, threshold=0.75, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        candidates = self.candidate_lst
        n = len(candidates)
        ids = [c.id for c in candidates]
        sim_matrix = np.zeros((n, n))

        # Compute Szymkiewicz–Simpson overlap coefficient
        for i in range(n):
            for j in range(n):
                if i == j:
                    sim_matrix[i][j] = 1.0
                    continue
                s1 = set(candidates[i].hit_indices or [])
                s2 = set(candidates[j].hit_indices or [])
                if not s1 or not s2:
                    sim = 0.0
                else:
                    sim = len(s1 & s2) / min(len(s1), len(s2))
                sim_matrix[i][j] = sim

        if verbose:
            df = pd.DataFrame(sim_matrix, index=ids, columns=ids)
            print("Overlap Coefficient Matrix:")
            print(df.round(2))

        # Build similarity graph
        G = nx.Graph()
        G.add_nodes_from(ids)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= threshold:
                    G.add_edge(ids[i], ids[j])

        if verbose:
            pos = nx.spring_layout(G, seed=452)
            fig, ax = plt.subplots(figsize=(8, 6))
            nx.draw(
                G, pos, ax=ax,
                with_labels=True,
                node_color='lightblue',
                edge_color='gray',
                node_size=800,
                font_size=10
            )
            ax.set_title(f"Candidate Similarity Graph (Threshold = {threshold:.2f})")
            plt.tight_layout()
            plt.show()

        # Keep strongest candidate in each group, absorb extra hit indices
        id_to_candidate = {c.id: c for c in candidates}
        dominant_candidates = []

        for group in nx.connected_components(G):
            group_cands = [id_to_candidate[i] for i in group]
            centrality = nx.degree_centrality(G)
            dominant = max(group_cands, key=lambda c: (centrality[c.id], c.hits))

            dominant_hits = set(dominant.hit_indices or [])
            absorbed_hits = set()

            for c in group_cands:
                if c is not dominant:
                    absorbed_hits.update(set(c.hit_indices or []) - dominant_hits)

            dominant.absorbed = sorted(absorbed_hits)
            dominant_candidates.append(dominant)

            if verbose and len(group_cands) > 1:
                group_ids = [c.id for c in group_cands]
                print(f"Merged group {group_ids} into candidate {dominant.id}. Absorbed {len(dominant.absorbed)} extra hit indices.")

        self.candidate_lst = dominant_candidates

        if verbose:
            print(f"Retaining {len(dominant_candidates)} dominant candidate{'s' if len(dominant_candidates) != 1 else ''} from similarity groups.")

    def get_max_start_index(self):
        max_start = int(self.num_peaks / 2)
        return min(max_start, self.num_peaks - 1)

    def get_min_hits(self):
        min_hits = math.ceil(sqrt(self.num_peaks))
        return max(min_hits, 1)

    def remove_outliers(self, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        min_hits = self.get_min_hits()

        original_count = len(self.candidate_lst)
        self.candidate_lst = [
            candidate
            for candidate in self.candidate_lst
            if len(candidate.hit_indices) + len(candidate.absorbed) >= min_hits
        ]

        if verbose:
            removed_count = original_count - len(self.candidate_lst)
            print(f"Removed {removed_count} outlier candidate{'s' if removed_count != 1 else ''} (min_hits = {min_hits})")


    def count_unused_peaks(self): # must be called after detect_regimes()
        used_peaks = set()
        for candidate in self.candidate_lst:
            for peak in candidate.hit_indices:
                used_peaks.add(peak)
            for peak in candidate.absorbed:
                used_peaks.add(peak)
        return self.num_peaks - len(used_peaks)

    def get_unused_peak_proportion(self):
        return self.count_unused_peaks() / self.num_peaks

    def get_num_regimes(self):
        return len(self.candidate_lst)

    def detect_regimes(self, verbose=None):
        verbose = self.verbose if verbose is None else verbose

        if verbose:
            print("Step 1: Generating candidates...")
        self.generate_candidates(verbose=verbose)

        if verbose:
            print("\nStep 2: Adding hit data...")
        self.add_hit_data(verbose=verbose)

        if verbose:
            print("\nStep 3: Pruning by binomial test...")
        self.prune_insufficient_hits(verbose=verbose)

        if verbose:
            print("\nStep 4: Grouping by hit similarity (overlap coefficient)...")
        self.group_candidates_by_similarity(verbose=verbose)

        if verbose:
            print("\nStep 5: Remove final outlier candidates with insufficient hits")
        self.remove_outliers(verbose=verbose)

        if verbose:
            print("\nFinal number of remaining candidates:")
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

    def __repr__(self):
        res = f"ID: {self.id}, d: {self.d:.6f}, Anchor: {self.anchor:.6f}"
        if self.hits and self.tries:
            res += f", Hits: {self.hits}, Tries: {self.tries}, Indices: {self.hit_indices}"
        if self.absorbed:
            res += f", Absorbed: {self.absorbed}"
        return res

    def count_hits(self, x, y, x_margin, y_margin):#, confidence=0.99):
        x = np.array(x)
        y = np.array(y)

        # Anchor is a guaranteed hit
        anchor_idx = np.where(x == self.anchor)[0][0]

        hit_indices = [int(anchor_idx)]
        tries = 1
        last_hit = self.anchor

        while True:
            t = last_hit + self.d
            if t > x[-1] + x_margin:
                break

            # Find candidate peaks within x-margin
            x_mask = (x >= t - x_margin) & (x <= t + x_margin)
            candidate_idxs = np.where(x_mask)[0]

            if candidate_idxs.size > 0:
                # Only keep candidates whose y-values are within y_margin of moving average y
                recent_hits_y = y[hit_indices[-3:]]  # last 3 hits
                expected_y = np.mean(recent_hits_y)
                y_diff = np.abs(y[candidate_idxs] - expected_y)
                valid_mask = y_diff <= y_margin
                valid_idxs = candidate_idxs[valid_mask]

                if valid_idxs.size > 0:
                    # Pick closest in x
                    distances = np.abs(x[valid_idxs] - t)
                    hit_idx = int(valid_idxs[np.argmin(distances)])
                    hit_indices.append(hit_idx)
                    last_hit = x[hit_idx]  # snap to actual hit
                else:
                    last_hit = t  # step forward linearly
            else:
                last_hit = t  # step forward linearly

            tries += 1

        self.hit_indices = hit_indices
        self.hits = len(hit_indices)
        self.tries = tries

    def binomial_test(self, p_null, alpha=0.01):
        if not self.hits:
            raise RuntimeError("Must call `count_hits()` before `binomial_test()`")
        if self.hits < 3:
            return False
        p_value = 1 - binom.cdf(self.hits - 2, self.tries - 2, p_null) # skip first 2 guaranteed hits
        return p_value < alpha


def plot_candidate_peak_sets(acceleration0, peaks, candidates):
    """
    Plot the acceleration signal with peaks, coloring each candidate's peak set uniquely.
    Shared peaks are plotted with small vertical jitter so all colors are visible.
    Absorbed peaks are plotted in the same color but lighter.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    peaks = np.array(peaks)
    cmap = plt.colormaps["Set1"]
    plt.figure(figsize=(10, 6))
    plt.plot(acceleration0, label='Acceleration 0', color='blue', alpha=0.7)

    # Map each peak index to all candidate indices using it
    peak_to_candidates = defaultdict(list)
    for i, candidate in enumerate(candidates):
        if candidate.hit_indices:
            for hit in candidate.hit_indices:
                peak_to_candidates[hit].append(i)

    used_peak_indices = set()

    for i, candidate in enumerate(candidates):
        color = cmap(i % cmap.N)

        # Plot main hit_indices with jitter
        if candidate.hit_indices:
            for hit in candidate.hit_indices:
                peak_idx = peaks[hit]
                base_y = acceleration0[peak_idx]
                shared_users = peak_to_candidates[hit]
                jitter = 0.01 * (shared_users.index(i) - len(shared_users) / 2)
                plt.scatter(peak_idx, base_y + jitter,
                            color=color,
                            label=f"Candidate {candidate.id}" if hit == candidate.hit_indices[0] else None,
                            zorder=3, marker="x", s=70)
                used_peak_indices.add(hit)

        # Plot absorbed peaks in lighter color
        if hasattr(candidate, 'absorbed') and candidate.absorbed:
            for hit in candidate.absorbed:
                if hit not in used_peak_indices:
                    peak_idx = peaks[hit]
                    plt.scatter(peak_idx, acceleration0[peak_idx],
                                color=color, alpha=0.3,
                                label=f"Absorbed (Cand {candidate.id})" if hit == candidate.absorbed[0] else None,
                                zorder=2, marker="x", s=60)
                    used_peak_indices.add(hit)

    # Unused peaks
    all_peak_indices = set(range(len(peaks)))
    unused_peak_indices = np.array(sorted(all_peak_indices - used_peak_indices))
    if len(unused_peak_indices) > 0:
        unused_peaks = peaks[unused_peak_indices]
        plt.scatter(unused_peaks, acceleration0[unused_peaks],
                    color='gray', alpha=0.5, label='Unused Peaks',
                    zorder=1, marker="x", s=40)

    plt.title("Acceleration Signal with Candidate-Colored Peaks (Shared = Jittered, Absorbed = Faded)")
    plt.xlabel("Time (index)")
    plt.ylabel("Acceleration")
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_boilings_data(x, y, run_length, verbose=False):
    if len(x) < 3: # require at least 3 peaks
        return 0, 0
    if len(x) > 300: # unreasonable run time; will yield many regimes
        return 3, 0
    candidates = Candidates(x, y, run_length)
    candidates.detect_regimes(verbose=verbose)
    num_boilings = candidates.get_num_regimes()
    unused_peak_proportion = candidates.get_unused_peak_proportion()
    return min(num_boilings, 3), unused_peak_proportion