
import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_repo_root(start=None):
    start = Path.cwd() if start is None else Path(start)

    candidates = [start]
    for parent in start.parents:
        candidates.append(parent)

    for path in candidates:
        if (path / "MultiplePFAnalysis" / "__init__.py").is_file():
            return path
    raise FileNotFoundError("Could not find repository root containing MultiplePFAnalysis")


REPO_ROOT = find_repo_root()

DEFAULT_INPUT_DIR = REPO_ROOT / "MultiplePFAnalysis" / "results" / "rotation_remapping_correlations"

DEFAULT_FIGURE_DIR = DEFAULT_INPUT_DIR / "figures"


def load_pair_data(csv_files):
    pair_data = {}

    for csv_path in csv_files:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pair_key = (row["exp_id_a"], row["exp_id_b"])
                rotation = int(row["rotation"])
                uid = (row["rat_id"], row["tetrode_nr"], row["cluster_id"])
                r = float(row["r"])

                if pair_key not in pair_data:
                    pair_data[pair_key] = {
                        "baseline_rotation": int(row["baseline_rotation"]),
                        "by_rotation": {},
                    }
                by_rotation = pair_data[pair_key]["by_rotation"]

                if rotation not in by_rotation:
                    by_rotation[rotation] = {}
                by_rotation[rotation][uid] = r

    return pair_data


def filter_to_one_rat(values_by_rotation, rat_id):
    filtered = {}
    for rotation, unit_values in values_by_rotation.items():
        filtered[rotation] = {}
        for uid, r in unit_values.items():
            if uid[0] != rat_id:
                continue
            filtered[rotation][(uid[1], uid[2])] = r
    return filtered


PAIR_LABELS_AND_BASELINE_ROTATION = [
    ("a_b", 90),
    ("b_c", 90),
    ("c_d", 90),
    ("a_c", 180),
    ("b_d", 180),
]

BLOB_COMPARISON_ROTATION = 0

DEFAULT_PERCENTILE = 10.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing per-rat *_rotation_correlations.csv files.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR / "blob_unit_overlap",
        help="Directory to save every plot as a PNG.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="How big the top slice of the actual diff values needs to be to "
        "count as rotationally remapped, e.g. 10 means the top 10%%.",
    )
    return parser.parse_args()


def short_env_name(experiment_id):
    if experiment_id.startswith("exp_scales_"):
        return experiment_id[len("exp_scales_"):]
    return experiment_id


def format_blob_label(label):
    return label.replace("_", " → ")


def find_pair_key_for_label(pair_data, label):
    for pair_key in pair_data:
        exp_id_a, exp_id_b = pair_key
        candidate_label = f"{short_env_name(exp_id_a)}_{short_env_name(exp_id_b)}"
        if candidate_label == label:
            return pair_key
    raise KeyError(f"No pair found for label {label!r} in the loaded CSVs")


def discover_rat_ids(pair_data):
    rat_ids = set()
    for pair_key in pair_data:
        by_rotation = pair_data[pair_key]["by_rotation"]
        for rotation in by_rotation:
            unit_values = by_rotation[rotation]
            for uid in unit_values:
                rat_ids.add(uid[0])
    return sorted(rat_ids)


def compute_percentile_threshold(diff_values, percentile):
    return float(np.percentile(diff_values, 100.0 - percentile))


def compute_blob_units(rat_results, baseline_rotation, comparison_rotation, percentile=DEFAULT_PERCENTILE):
    baseline_units = rat_results[baseline_rotation]
    compare_units = rat_results[comparison_rotation]

    common_ids = set()
    for uid in baseline_units:
        if uid in compare_units:
            common_ids.add(uid)

    differences = {}
    for uid in common_ids:
        b = baseline_units[uid]
        c = compare_units[uid]
        if not np.isnan(b) and not np.isnan(c):
            differences[uid] = b - c

    if len(differences) == 0:
        return set(), differences, float("nan")

    diff_values = []
    for uid in differences:
        diff_values.append(differences[uid])
    diff_values = np.array(diff_values)

    cutoff = compute_percentile_threshold(diff_values, percentile)

    blob_ids = set()
    for uid in differences:
        if differences[uid] >= cutoff:
            blob_ids.add(uid)

    return blob_ids, differences, cutoff


MAX_SAVEFIG_ATTEMPTS = 5
SECONDS_BETWEEN_SAVEFIG_RETRIES = 0.3


def save_current_figure(figure_path):
    for attempt in range(1, MAX_SAVEFIG_ATTEMPTS + 1):
        try:
            plt.savefig(figure_path, dpi=150, bbox_inches="tight")
            return
        except OSError as save_error:
            if attempt == MAX_SAVEFIG_ATTEMPTS:
                raise
            print(f"  Could not save {figure_path} (attempt {attempt}: {save_error}), retrying...")
            time.sleep(SECONDS_BETWEEN_SAVEFIG_RETRIES)


def title_with_threshold_info(title, threshold_label):
    if threshold_label is None:
        return title
    return f"{title}\n({threshold_label})"


def format_threshold_label(percentile=DEFAULT_PERCENTILE):
    return f"top {percentile:g}% percentile (empirical, fit separately per rat and pair)"


def compute_membership_counts(pair_labels, blob_sets_by_label, blob_candidates_by_label):
    shared_candidates = None
    for label in pair_labels:
        candidates = blob_candidates_by_label[label]
        if shared_candidates is None:
            shared_candidates = set(candidates)
        else:
            shared_candidates &= candidates
    if shared_candidates is None:
        shared_candidates = set()

    counts = {}
    for uid in shared_candidates:
        membership_list = []
        for label in pair_labels:
            membership_list.append(uid in blob_sets_by_label[label])
        membership = tuple(membership_list)
        counts[membership] = counts.get(membership, 0) + 1
    return counts


def compute_degree_histogram(pooled_combination_counts, n_labels):
    histogram = {}
    for degree in range(n_labels + 1):
        histogram[degree] = 0
    for membership, count in pooled_combination_counts.items():
        degree = 0
        for value in membership:
            if value:
                degree += 1
        histogram[degree] += count
    return histogram


def plot_blob_degree_histogram(histogram, n_labels, figure_path=None, threshold_label=None):
    degrees = list(range(n_labels + 1))
    bar_heights = []
    for degree in degrees:
        bar_heights.append(histogram[degree])
    max_height = max(bar_heights)
    if max_height == 0:
        max_height = 1

    bar_colors = []
    for degree in degrees:
        if degree == n_labels:
            bar_colors.append("#e34948")
        else:
            bar_colors.append("#2a78d6")

    plt.figure(figsize=(8, 5))
    plt.bar(degrees, bar_heights, color=bar_colors)
    for degree, height in zip(degrees, bar_heights):
        plt.text(degree, height + max_height * 0.01, str(height), ha="center", va="bottom", fontsize=9)

    plt.xticks(degrees)
    plt.xlabel(f"Number of the {n_labels} comparisons a unit was rotationally remapped in")
    plt.ylabel("Number of units (pooled across rats)")
    plt.title(title_with_threshold_info(
        "Rotationally remapped units by number of comparisons involved (pooled across rats)",
        threshold_label,
    ))
    plt.tight_layout()

    if figure_path is not None:
        save_current_figure(figure_path)

    plt.close()


def plot_blob_degree_histogram_mean_across_rats(degree_counts_by_rat, n_labels, figure_path=None, threshold_label=None):
    degrees = list(range(n_labels + 1))
    bar_heights = []
    bar_errors = []
    for degree in degrees:
        counts = degree_counts_by_rat[degree]
        bar_heights.append(float(np.mean(counts)))
        bar_errors.append(float(np.std(counts)))

    top_of_data_by_bar = []
    for degree in degrees:
        counts = degree_counts_by_rat[degree]
        highest_point = bar_heights[degree] + bar_errors[degree]
        for value in counts:
            if value > highest_point:
                highest_point = value
        top_of_data_by_bar.append(highest_point)

    overall_top = 0.0
    for value in top_of_data_by_bar:
        if value > overall_top:
            overall_top = value
    margin = overall_top * 0.12 + 0.01

    bar_colors = []
    for degree in degrees:
        if degree == n_labels:
            bar_colors.append("#e34948")
        else:
            bar_colors.append("#2a78d6")

    plt.figure(figsize=(8, 5))
    plt.bar(degrees, bar_heights, yerr=bar_errors, capsize=4, color=bar_colors, alpha=0.85)

    for degree in degrees:
        counts = degree_counts_by_rat[degree]
        jitter_x = []
        for value in counts:
            jitter_x.append(degree)
        plt.scatter(jitter_x, counts, color="#0b0b0b", zorder=3, s=20)

    for degree in degrees:
        mean_label_y = top_of_data_by_bar[degree] + margin * 0.3
        plt.text(degree, mean_label_y, f"{bar_heights[degree]:.1f}", ha="center", va="bottom", fontsize=8)

    plt.ylim(bottom=0, top=overall_top + margin * 2)

    plt.xticks(degrees)
    plt.xlabel(f"Number of the {n_labels} comparisons a unit was rotationally remapped in")
    plt.ylabel("Number of units per rat (mean ± SD across rats)")
    plt.title(title_with_threshold_info(
        "Rotationally remapped units by number of comparisons involved (mean ± SD across rats)",
        threshold_label,
    ))
    plt.tight_layout()

    if figure_path is not None:
        save_current_figure(figure_path)

    plt.close()


def main():
    args = parse_args()

    csv_files = sorted(args.input_dir.glob("*_rotation_correlations.csv"))

    pair_data = load_pair_data(csv_files)
    rat_ids = discover_rat_ids(pair_data)
    print(f"Found {len(rat_ids)} rats: {', '.join(rat_ids)}")

    pair_key_for_label = {}
    baseline_rotation_for_label = {}
    for label, baseline_rotation in PAIR_LABELS_AND_BASELINE_ROTATION:
        pair_key_for_label[label] = find_pair_key_for_label(pair_data, label)
        baseline_rotation_for_label[label] = baseline_rotation

    pair_labels = []
    for label, _ in PAIR_LABELS_AND_BASELINE_ROTATION:
        pair_labels.append(label)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {args.figure_dir}")

    pooled_combination_counts = {}

    degree_counts_by_rat = {}
    for degree in range(len(pair_labels) + 1):
        degree_counts_by_rat[degree] = []

    for rat_id in rat_ids:
        print(f"\n=== Rat {rat_id} ===")

        blob_sets_by_label = {}
        blob_candidates_by_label = {}

        for label in pair_labels:
            pair_key = pair_key_for_label[label]
            baseline_rotation = baseline_rotation_for_label[label]
            rat_results = filter_to_one_rat(pair_data[pair_key]["by_rotation"], rat_id)

            blob_ids, differences, cutoff = compute_blob_units(
                rat_results, baseline_rotation, BLOB_COMPARISON_ROTATION, args.percentile,
            )
            blob_sets_by_label[label] = blob_ids
            blob_candidates_by_label[label] = set(differences.keys())

            print(
                f"[{rat_id}] {format_blob_label(label)}: {len(blob_ids)} rotationally "
                f"remapped units (of {len(differences)} candidates, threshold diff >= {cutoff:.3f})"
            )

        this_rat_membership_counts = compute_membership_counts(pair_labels, blob_sets_by_label, blob_candidates_by_label)
        for membership, count in this_rat_membership_counts.items():
            pooled_combination_counts[membership] = pooled_combination_counts.get(membership, 0) + count

        this_rat_degree_histogram = compute_degree_histogram(this_rat_membership_counts, len(pair_labels))
        for degree in range(len(pair_labels) + 1):
            degree_counts_by_rat[degree].append(this_rat_degree_histogram[degree])

    summary_threshold_label = format_threshold_label(percentile=args.percentile)

    degree_histogram = compute_degree_histogram(pooled_combination_counts, len(pair_labels))

    degree_figure_path = args.figure_dir / "blob_unit_degree_histogram.png"
    plot_blob_degree_histogram(
        degree_histogram, len(pair_labels), figure_path=degree_figure_path,
        threshold_label=summary_threshold_label,
    )
    print(f"Saved degree-histogram plot to {degree_figure_path}")

    degree_mean_figure_path = args.figure_dir / "blob_unit_degree_histogram_mean_across_rats.png"
    plot_blob_degree_histogram_mean_across_rats(
        degree_counts_by_rat, len(pair_labels), figure_path=degree_mean_figure_path,
        threshold_label=summary_threshold_label,
    )
    print(f"Saved degree-histogram (mean across rats) plot to {degree_mean_figure_path}")


if __name__ == "__main__":
    main()
