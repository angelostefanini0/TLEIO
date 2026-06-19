from pathlib import Path
from collections import OrderedDict
import argparse

import yaml
import numpy as np
import matplotlib.pyplot as plt


SEQUENCES = [
    "ME000", "ME001", "ME002", "ME003", "ME004", "ME005", "ME006", "ME007",
    "MH000", "MH001", "MH002", "MH003", "MH004", "MH005", "MH006", "MH007",
]

# YAML key -> plot label
METHODS = OrderedDict([
    ("TLIO", "3D RoNIN"),
    ("Eventsformer_no_vggt", "EventsFormer w/o RoPE"),
    ("Eventsformer_vggt", "EventsFormer w/ RoPE"),
    ("TLEIO", "TLEIO"),
    ("TLEIO_no_cov", "TLEIO no cov"),
])

# Original colors
STYLE = {
    "3D RoNIN": dict(color="#5DA5DA", linestyle="-", linewidth=2.4),
    "EventsFormer w/o RoPE": dict(color="#60BD68", linestyle="-", linewidth=2.4),
    "EventsFormer w/ RoPE": dict(color="#F15854", linestyle="-", linewidth=2.4),
    "TLEIO": dict(color="#D99A00", linestyle="-", linewidth=2.8),
    "TLEIO no cov": dict(color="#8A2BE2", linestyle="-", linewidth=2.4),
}


def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def load_results(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if "experiments_ate_metrics" not in raw:
        raise KeyError("Missing top-level key: experiments_ate_metrics")

    metrics = raw["experiments_ate_metrics"]
    data = OrderedDict()

    for yaml_key, label in METHODS.items():
        if yaml_key not in metrics:
            raise KeyError(f"Missing method in YAML: {yaml_key}")

        results = metrics[yaml_key].get("results", {})
        values = []

        for seq in SEQUENCES:
            if seq not in results:
                raise KeyError(f"Missing sequence {seq} for method {yaml_key}")
            values.append(float(results[seq]))

        data[label] = np.asarray(values, dtype=float)

    return data


def ecdf(values, xmax):
    values = np.sort(np.clip(values, 0.0, xmax))
    n = len(values)

    xs = [0.0]
    ys = [0.0]

    for i, x in enumerate(values):
        prev_y = i / n
        new_y = (i + 1) / n
        xs.extend([x, x])
        ys.extend([prev_y, new_y])

    xs.append(xmax)
    ys.append(1.0)

    return np.asarray(xs), 100.0 * np.asarray(ys)


def normalized_auc(values, xmax):
    xs, ys = ecdf(values, xmax)
    ys = ys / 100.0
    return float(np.trapz(ys, xs) / xmax)


def sorted_by_auc(data, methods, auc_xmax):
    return sorted(
        methods,
        key=lambda method: normalized_auc(data[method], auc_xmax),
        reverse=True,
    )


def configure_axes(ax, xmax):
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, 105.0)

    if xmax <= 4.0:
        ax.set_xticks([0, 1, 2, 3, 4])
    else:
        ax.set_xticks([0, 2, 4, 6, 8, 10])

    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.set_xlabel("ATE (m)")
    ax.set_ylabel("runs(%)")

    # RAMPVO-like style
    ax.grid(True, axis="y", color="0.75", linewidth=0.6)
    ax.grid(False, axis="x")

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(axis="both", which="major", pad=2)


def draw_panel(ax, data, methods, xmax, auc_xmax):
    methods = sorted_by_auc(data, methods, auc_xmax)

    for method in methods:
        xs, ys = ecdf(data[method], xmax)
        auc = normalized_auc(data[method], auc_xmax)
        ax.plot(xs, ys, label=f"{method}  AUC={auc:.2f}", **STYLE[method])

    configure_axes(ax, xmax=xmax)

    leg = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.7",
        borderpad=0.30,
        handlelength=2.1,
        handletextpad=0.45,
        labelspacing=0.22,
    )
    leg.get_frame().set_linewidth(0.5)


def save_figure(fig, out_base: Path):
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_full_comparison(data, out_base):
    fig, ax = plt.subplots(figsize=(4.15, 3.15))

    draw_panel(
        ax,
        data,
        ["3D RoNIN", "EventsFormer w/o RoPE", "EventsFormer w/ RoPE", "TLEIO"],
        xmax=11.0,
        auc_xmax=11.0,
    )

    fig.subplots_adjust(left=0.15, right=0.985, top=0.985, bottom=0.16)
    save_figure(fig, out_base)


def plot_cov_ablation(data, out_base):
    fig, ax = plt.subplots(figsize=(4.15, 3.15))

    draw_panel(
        ax,
        data,
        ["TLEIO", "TLEIO no cov"],
        xmax=4.0,
        auc_xmax=11.0,
    )

    fig.subplots_adjust(left=0.15, right=0.985, top=0.985, bottom=0.16)
    save_figure(fig, out_base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--out-dir", default="figures/tartanair_ate_cdf_rampvo_style")
    args = parser.parse_args()

    setup_matplotlib()

    yaml_path = Path(args.yaml)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(yaml_path)

    plot_full_comparison(
        data,
        out_dir / "cdf_full_comparison_rampvo_style",
    )

    plot_cov_ablation(
        data,
        out_dir / "cdf_covariance_ablation_rampvo_style",
    )

    print(f"Loaded YAML: {yaml_path.resolve()}")
    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
