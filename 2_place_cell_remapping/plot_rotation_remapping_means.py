
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def find_repo_root(start=None):
    if start is None:
        start = Path.cwd()
    else:
        start = Path(start)

    candidates = [start]
    for parent in start.parents:
        candidates.append(parent)

    for path in candidates:
        if (path / "MultiplePFAnalysis" / "__init__.py").is_file():
            return path
    raise FileNotFoundError("Could not find repository root containing MultiplePFAnalysis")


REPO_ROOT = find_repo_root()

DEFAULT_INPUT_DIR = REPO_ROOT / "MultiplePFAnalysis" / "results" / "rotation_remapping_correlations"

DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "fisher_z_ttest"

DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_DIR / "figures"


def format_environment_label(experiment_id):
    if experiment_id.startswith("exp_scales_"):
        return f"Environment {experiment_id[len('exp_scales_'):]}"
    return experiment_id


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


def fisher_z(r_values):
    r_array = np.clip(np.asarray(r_values, dtype=float), -0.999, 0.999)
    return np.arctanh(r_array)


def mean_r_via_fisher_z(r_values):
    return float(np.tanh(np.mean(fisher_z(r_values))))


def compute_rat_means_by_rotation(values_by_rotation):
    means_by_rotation = {}
    for rotation, unit_values in values_by_rotation.items():
        r_values_by_rat = {}
        for uid, r in unit_values.items():
            rat_id = uid[0]
            if rat_id not in r_values_by_rat:
                r_values_by_rat[rat_id] = []
            r_values_by_rat[rat_id].append(r)

        means_by_rotation[rotation] = {}
        for rat_id, r_values in r_values_by_rat.items():
            means_by_rotation[rotation][rat_id] = mean_r_via_fisher_z(r_values)

    return means_by_rotation


def stars_for_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def log(message, stats_file=None):
    print(message)
    if stats_file is not None:
        print(message, file=stats_file)


def control_significance_of_rotation(pair_results, pair_label="", baseline_rotation=90, comparison_rotations=(0, 180, 270), entity_name="units", stats_file=None):

    all_rotations = (0, 90, 180, 270)

    all_sets = []

    for rot in all_rotations:
        unit_ids = set(pair_results[rot].keys())
        all_sets.append(unit_ids)

    common_ids = all_sets[0]

    for s in all_sets[1:]:
        common_ids = common_ids.intersection(s)

    filtered_ids = []

    for uid in common_ids:

        has_nan = False

        for rot in all_rotations:
            value = pair_results[rot][uid]

            if np.isnan(value):
                has_nan = True
                break

        if not has_nan:
            filtered_ids.append(uid)

    log(f"[{pair_label}] n {entity_name} for statistical test: {len(filtered_ids)}", stats_file)
    log(f"[{pair_label}] n {entity_name} before dropping NaNs: {len(common_ids)}", stats_file)

    r_by_rot = {}

    for rot in all_rotations:

        values = []

        for uid in filtered_ids:
            values.append(pair_results[rot][uid])

        r_by_rot[rot] = np.array(values)

    r_0 = r_by_rot[0]
    r_90 = r_by_rot[90]
    r_180 = r_by_rot[180]
    r_270 = r_by_rot[270]

    z_by_rot = {}
    for rot in all_rotations:
        z_by_rot[rot] = fisher_z(r_by_rot[rot])

    z_baseline = z_by_rot[baseline_rotation]
    n = len(filtered_ids)
    t_crit = stats.t.ppf(0.975, n - 1)

    t_p = []
    t_list = []
    mean_diff_list = []
    ci_low_list = []
    ci_high_list = []

    for rot in comparison_rotations:

        z_compare = z_by_rot[rot]

        t, p_two = stats.ttest_rel(z_baseline, z_compare)
        t_p.append(p_two)
        t_list.append(t)

        diffs = z_baseline - z_compare
        mean_diff = np.mean(diffs)
        sem_diff = np.std(diffs, ddof=1) / np.sqrt(n)
        mean_diff_list.append(mean_diff)
        ci_low_list.append(mean_diff - t_crit * sem_diff)
        ci_high_list.append(mean_diff + t_crit * sem_diff)

    _, t_pvals_corrected, _, _ = multipletests(t_p, alpha=0.05, method="holm")

    for i in range(len(comparison_rotations)):
        rot = comparison_rotations[i]
        t = t_list[i]
        p_two = t_p[i]
        p_two_corr = t_pvals_corrected[i]

        log(
            f"[{pair_label}] paired t-test (Fisher z) {baseline_rotation}° vs {rot}°: "
            f"t({n - 1})={t:.3f}, p_raw={p_two:.4g}, p_holm={p_two_corr:.4g}, "
            f"mean dz={mean_diff_list[i]:.3f}, 95% CI [{ci_low_list[i]:.3f}, {ci_high_list[i]:.3f}]",
            stats_file,
        )

    log(
        f"[{pair_label}] Medians (r)\n"
        f"  0°   : {np.median(r_0):.3f}\n"
        f"  90°  : {np.median(r_90):.3f}\n"
        f" 180°  : {np.median(r_180):.3f}\n"
        f" 270°  : {np.median(r_270):.3f}",
        stats_file,
    )

    log(
        f"[{pair_label}] Means (averaged in Fisher z, back-transformed to r)\n"
        f"  0°   : {mean_r_via_fisher_z(r_0):.3f}\n"
        f"  90°  : {mean_r_via_fisher_z(r_90):.3f}\n"
        f" 180°  : {mean_r_via_fisher_z(r_180):.3f}\n"
        f" 270°  : {mean_r_via_fisher_z(r_270):.3f}",
        stats_file,
    )

    t_holm_by_rotation = {}
    for i in range(len(comparison_rotations)):
        rot = comparison_rotations[i]
        t_holm_by_rotation[rot] = t_pvals_corrected[i]

    return r_by_rot, filtered_ids, t_holm_by_rotation


def plot_rotation_scatter(values_by_rotation, rotation_x=0, rotation_y=90, pair_label="", figure_path=None):
    unit_ids_x = set(values_by_rotation[rotation_x].keys())
    unit_ids_y = set(values_by_rotation[rotation_y].keys())

    common_ids = unit_ids_x.intersection(unit_ids_y)

    filtered_ids = []

    for uid in common_ids:

        value_x = values_by_rotation[rotation_x][uid]
        value_y = values_by_rotation[rotation_y][uid]

        if not np.isnan(value_x) and not np.isnan(value_y):
            filtered_ids.append(uid)

    corr_x = []
    corr_y = []
    rat_ids = []

    for uid in filtered_ids:
        corr_x.append(values_by_rotation[rotation_x][uid])
        corr_y.append(values_by_rotation[rotation_y][uid])
        rat_ids.append(uid[0])

    corr_x = np.array(corr_x)
    corr_y = np.array(corr_y)
    rat_ids = np.array(rat_ids)

    plt.figure(figsize=(5, 5))

    unique_rat_ids = sorted(set(rat_ids))
    colors = plt.get_cmap("tab10")
    for i, rat_id in enumerate(unique_rat_ids):
        is_this_rat = rat_ids == rat_id
        plt.scatter(corr_x[is_this_rat], corr_y[is_this_rat], alpha=0.6, color=colors(i), label=rat_id)
    plt.legend(loc="upper left", fontsize=8, title="Rat")

    plt.plot([-1, 1], [-1, 1], 'k--', lw=1)
    plt.xlabel(f'Correlation ({rotation_x}°)')
    plt.ylabel(f'Correlation ({rotation_y}°)')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(f"{pair_label}\n{rotation_x}° vs {rotation_y}°")
    plt.tight_layout()

    if figure_path is not None:
        plt.savefig(figure_path, dpi=150)
        print(f"  Saved {figure_path}")

    plt.show()


def plot_combined_histogram_across_rats(values_by_rotation, pair_label="", figure_path=None, n_bins=10):
    rotations = [0, 90, 180, 270]
    colors = {0: "tab:blue", 90: "tab:orange", 180: "tab:green", 270: "tab:red"}

    bin_edges = np.linspace(-1, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    mean_proportions = {}
    all_rat_ids = set()

    for rotation in rotations:
        unit_values = values_by_rotation.get(rotation, {})

        r_values_by_rat = {}
        for uid, r in unit_values.items():
            if np.isnan(r):
                continue
            rat_id = uid[0]
            if rat_id not in r_values_by_rat:
                r_values_by_rat[rat_id] = []
            r_values_by_rat[rat_id].append(r)

        all_rat_ids.update(r_values_by_rat.keys())

        proportions_per_rat = []
        for rat_id, r_values in r_values_by_rat.items():
            counts, _ = np.histogram(r_values, bins=bin_edges)
            proportions_per_rat.append(counts / len(r_values))

        if proportions_per_rat:
            mean_proportions[rotation] = np.mean(proportions_per_rat, axis=0)
        else:
            mean_proportions[rotation] = np.zeros(n_bins)

    n_rotations = len(rotations)
    group_width = bin_width * 0.9
    bar_width = group_width / n_rotations

    plt.figure()
    for i, rotation in enumerate(rotations):
        offset = (i - (n_rotations - 1) / 2) * bar_width
        plt.bar(
            bin_centers + offset,
            mean_proportions[rotation],
            width=bar_width,
            color=colors[rotation],
            label=f"{rotation}°",
            edgecolor="k",
        )

    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Mean proportion of cells (across rats)")
    plt.xlim(-1, 1)
    plt.legend(title="Rotation")
    plt.title(f"{pair_label}\nAll Rotations (mean across {len(all_rat_ids)} rats)")
    plt.tight_layout()

    if figure_path is not None:
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {figure_path}")

    plt.show()


def plot_mean_correlation_by_rotation(values_by_rotation, pair_label="", baseline_rotation=90, t_holm_by_rotation=None, figure_path=None):
    rotations = [0, 90, 180, 270]
    colors = {0: "tab:blue", 90: "tab:orange", 180: "tab:green", 270: "tab:red"}

    rat_means_by_rotation = compute_rat_means_by_rotation(values_by_rotation)

    bar_means = []
    bar_err_low = []
    bar_err_high = []
    all_rat_ids = set()

    for rotation in rotations:
        values = np.array(list(rat_means_by_rotation[rotation].values()))
        all_rat_ids.update(rat_means_by_rotation[rotation].keys())
        z_values = fisher_z(values)
        mean_z = np.mean(z_values)
        sd_z = np.std(z_values, ddof=1)
        mean_r = np.tanh(mean_z)
        bar_means.append(mean_r)
        bar_err_low.append(mean_r - np.tanh(mean_z - sd_z))
        bar_err_high.append(np.tanh(mean_z + sd_z) - mean_r)

    x_positions = np.arange(len(rotations))

    bar_colors = []
    for rotation in rotations:
        bar_colors.append(colors[rotation])

    plt.figure(figsize=(6, 5))
    plt.bar(x_positions, bar_means, yerr=[bar_err_low, bar_err_high], capsize=5, color=bar_colors, edgecolor="k")

    for i, rotation in enumerate(rotations):
        values = list(rat_means_by_rotation[rotation].values())
        plt.scatter([x_positions[i]] * len(values), values, facecolors="white", edgecolors="black", zorder=3)

    show_stats = t_holm_by_rotation is not None

    if show_stats:
        baseline_x = x_positions[rotations.index(baseline_rotation)]

        bar_tops = []
        bar_bottoms = []
        for i in range(len(bar_means)):
            bar_tops.append(bar_means[i] + bar_err_high[i])
            bar_bottoms.append(bar_means[i] - bar_err_low[i])
        top = max(bar_tops)
        bottom = min(bar_bottoms)
        step = (top - bottom) * 0.12

        for i, rotation in enumerate(t_holm_by_rotation.keys()):
            compare_x = x_positions[rotations.index(rotation)]
            y = top + step * (i + 1)
            plt.plot([baseline_x, baseline_x, compare_x, compare_x], [y, y + step * 0.2, y + step * 0.2, y], color="black", linewidth=1)
            label = stars_for_p(t_holm_by_rotation[rotation])
            plt.text((baseline_x + compare_x) / 2, y + step * 0.2, label, ha="center", va="bottom", fontsize=8)

    x_tick_labels = []
    for rotation in rotations:
        x_tick_labels.append(f"{rotation}°")
    plt.xticks(x_positions, x_tick_labels)
    plt.xlabel("Rotation")
    plt.ylabel("Mean Correlation Coefficient (r)")
    plt.title(f"{pair_label}\nMean correlation by rotation (± SD in Fisher z, {len(all_rat_ids)} rats)")
    if show_stats:
        plt.figtext(0.5, 0.045, "vs. baseline: paired t-test on Fisher z rat means, Holm-corrected.", ha="center", fontsize=8)
        plt.figtext(0.5, 0.01, "* p<0.05, ** p<0.01, *** p<0.001, n.s. not significant", ha="center", fontsize=8)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
    else:
        plt.tight_layout()

    if figure_path is not None:
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {figure_path}")

    plt.show()


def main():
    input_dir = DEFAULT_INPUT_DIR
    figure_dir = DEFAULT_FIGURE_DIR
    stats_path = DEFAULT_OUTPUT_DIR / "rotation_remapping_stats_means.txt"

    csv_files = sorted(input_dir.glob("*_rotation_correlations.csv"))

    pair_data = load_pair_data(csv_files)

    figure_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {figure_dir}")

    stats_path.parent.mkdir(parents=True, exist_ok=True)

    all_rotations = (0, 90, 180, 270)

    with stats_path.open("w", encoding="utf-8") as stats_file:
        for pair_key, data in pair_data.items():
            exp_id_a, exp_id_b = pair_key
            baseline_rotation = data["baseline_rotation"]
            by_rotation = data["by_rotation"]
            pair_label = f"{format_environment_label(exp_id_a)} -> {format_environment_label(exp_id_b)}"
            safe_pair = f"{exp_id_a}_to_{exp_id_b}".replace("/", "_").replace("\\", "_")

            comparison_rotations = []
            for rotation in all_rotations:
                if rotation != baseline_rotation:
                    comparison_rotations.append(rotation)

            combined_hist_figure_path = figure_dir / f"{safe_pair}_all_rotations_mean_histogram.png"
            plot_combined_histogram_across_rats(by_rotation, pair_label=pair_label, figure_path=combined_hist_figure_path)

            _, _, t_holm_by_rotation = control_significance_of_rotation(
                compute_rat_means_by_rotation(by_rotation),
                pair_label=f"{pair_label} - rat means",
                baseline_rotation=baseline_rotation,
                comparison_rotations=comparison_rotations,
                entity_name="rats",
                stats_file=stats_file,
            )

            mean_by_rotation_figure_path = figure_dir / f"{safe_pair}_mean_correlation_by_rotation.png"
            plot_mean_correlation_by_rotation(
                by_rotation, pair_label=pair_label, baseline_rotation=baseline_rotation,
                t_holm_by_rotation=t_holm_by_rotation,
                figure_path=mean_by_rotation_figure_path,
            )

            for comparison_rotation in comparison_rotations:
                scatter_figure_path = figure_dir / f"{safe_pair}_{baseline_rotation}_vs_{comparison_rotation}_scatter.png"
                plot_rotation_scatter(by_rotation, baseline_rotation, comparison_rotation, pair_label, scatter_figure_path)

    print(f"Saved statistical-test output to {stats_path}")


if __name__ == "__main__":
    main()
