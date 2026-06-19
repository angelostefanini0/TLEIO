from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("figures/tartanair_ate_cdf")
X_MAX_AUC = 11.0

DATA = {
    "sequence": [
        "ME000", "ME001", "ME002", "ME003", "ME004", "ME005", "ME006", "ME007",
        "MH000", "MH001", "MH002", "MH003", "MH004", "MH005", "MH006", "MH007",
    ],
    "TLIO": [
        3.915, 7.962, 3.193, 5.831, 5.344, 2.192, 4.935, 2.616,
        10.848, 2.376, 3.101, 3.074, 2.141, 2.714, 3.362, 4.160,
    ],
    "EventsFormer w/o RoPE": [
        4.329, 1.818, 3.977, 5.389, 3.946, 1.784, 4.696, 2.338,
        4.069, 2.127, 4.128, 2.170, 2.036, 2.366, 1.883, 3.444,
    ],
    "EventsFormer w/ RoPE": [
        3.5913, 2.7744, 3.4298, 4.3372, 2.4225, 2.6090, 2.9120, 1.2759,
        4.0477, 1.7551, 4.5339, 1.8162, 1.8216, 3.4524, 2.0499, 3.4678,
    ],
    "TLEIO": [
        2.13, 1.20, 1.45, 1.37, 1.37, 1.19, 0.48, 0.72,
        2.65, 0.14, 1.16, 0.22, 0.60, 0.48, 1.15, 1.44,
    ],
    "TLEIO no cov": [
        2.216127, 1.571009, 1.340590, 1.974953, 1.569339, 1.431087, 1.048662, 0.888658,
        2.769894, 0.398785, 2.119496, 0.216452, 1.346663, 0.767626, 1.132565, 1.887184,
    ],
}

STYLE = {
    "TLIO": {"color": "#5DA5DA", "linestyle": "-", "marker": "o"},
    "EventsFormer w/o RoPE": {"color": "#60BD68", "linestyle": "-", "marker": "s"},
    "EventsFormer w/ RoPE": {"color": "#F15854", "linestyle": "-", "marker": "^"},
    "TLEIO": {"color": "#F2CF5B", "linestyle": "-", "marker": "D"},
    "TLEIO no cov": {"color": "#8C8C8C", "linestyle": "--", "marker": "v"},
}

SHORT_LABEL = {
    "TLIO": "TLIO",
    "EventsFormer w/o RoPE": "w/o RoPE",
    "EventsFormer w/ RoPE": "w/ RoPE",
    "TLEIO": "TLEIO",
    "TLEIO no cov": "TLEIO-noCov",
}


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def normalized_auc(values: np.ndarray, x_max: float = X_MAX_AUC) -> float:
    x, y = empirical_cdf(values)
    x_clip = np.clip(x, 0.0, x_max)
    x_step = np.concatenate([[0.0], x_clip, [x_max]])
    y_step = np.concatenate([[0.0], y, [1.0]])
    auc = np.trapz(y_step, x_step)
    return float(auc / x_max)


def mean_ate(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def auc_anchor(method: str, count: int, xmax: float) -> tuple[float, float]:
    if xmax <= 4.0:
        slots = {
            "TLEIO": (0.12, 0.88),
            "TLEIO no cov": (0.55, 0.64),
        }
    elif count >= 4:
        slots = {
            "TLEIO": (0.08, 0.92),
            "EventsFormer w/ RoPE": (0.08, 0.72),
            "EventsFormer w/o RoPE": (0.31, 0.55),
            "TLIO": (0.50, 0.38),
        }
    elif count == 3:
        slots = {
            "EventsFormer w/ RoPE": (0.08, 0.88),
            "EventsFormer w/o RoPE": (0.31, 0.66),
            "TLIO": (0.50, 0.44),
        }
    else:
        slots = {
            "EventsFormer w/o RoPE": (0.12, 0.84),
            "TLIO": (0.50, 0.52),
        }
    return slots.get(method, (0.10, 0.85))


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.45,
        "lines.linewidth": 1.7,
    })


def plot_cdf(
    df: pd.DataFrame,
    methods: list[str],
    output_stem: str,
    title: str,
    xlim: tuple[float, float],
    figsize: tuple[float, float] = (3.35, 2.35),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    methods = sorted(methods, key=lambda method: normalized_auc(df[method].to_numpy(dtype=float)), reverse=True)

    for idx, method in enumerate(methods):
        values = df[method].to_numpy(dtype=float)
        x, y = empirical_cdf(values)
        ax.step(x, y, where="post", label=SHORT_LABEL.get(method, method), linewidth=2.3, **STYLE[method])
        ax.scatter(x, y, s=13, zorder=3, color=STYLE[method]["color"], marker=STYLE[method]["marker"])
        ax.text(
            *auc_anchor(method, len(methods), xlim[1]),
            f"AUC = {normalized_auc(values):.2f}",
            transform=ax.transAxes,
            color=STYLE[method]["color"],
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("ATE threshold [m]")
    ax.set_ylabel("Fraction of sequences")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, borderpad=0.45)
    fig.tight_layout(pad=0.35)

    for suffix in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{output_stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame) -> None:
    rows = []
    for method in DATA.keys():
        if method == "sequence":
            continue
        values = df[method].to_numpy(dtype=float)
        rows.append({
            "method": method,
            "mean_ate_m": mean_ate(values),
            "median_ate_m": float(np.median(values)),
            "auc_norm_xmax_11m": normalized_auc(values),
            "best_sequence_count": int(np.sum(values == df.drop(columns=["sequence"]).min(axis=1).to_numpy())),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    df.to_csv(OUT_DIR / "ate_values.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    df = pd.DataFrame(DATA)
    write_summary(df)

    plot_cdf(
        df,
        ["TLIO", "EventsFormer w/o RoPE"],
        "cdf_01_tlio_vs_eventsformer_no_rope",
        "Base comparison",
        xlim=(0.0, 11.0),
    )
    plot_cdf(
        df,
        ["TLIO", "EventsFormer w/o RoPE", "EventsFormer w/ RoPE"],
        "cdf_02_rotary_position_embedding",
        "Rotary Position Embedding",
        xlim=(0.0, 11.0),
    )
    plot_cdf(
        df,
        ["TLIO", "EventsFormer w/o RoPE", "EventsFormer w/ RoPE", "TLEIO"],
        "cdf_03_full_comparison",
        "Full comparison",
        xlim=(0.0, 11.0),
        figsize=(3.55, 2.45),
    )
    plot_cdf(
        df,
        ["TLEIO", "TLEIO no cov"],
        "cdf_04_covariance_ablation_zoom",
        "Covariance ablation",
        xlim=(0.0, 4.0),
    )

    print(f"Saved plots and CSV files to {OUT_DIR}")


if __name__ == "__main__":
    main()
