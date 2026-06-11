from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
    }
)

COLORS = {
    "baseline": "#4D4D4D",
    "baseline_eqstep": "#6C8EBF",
    "hardneg": "#2E9E44",
    "hardneg_sched": "#8E6AAD",
    "formal_main": "#4C78A8",
    "candidate_verified": "#E58932",
    "planned": "#C7C7C7",
    "blocked": "#D95F5F",
    "not_run": "#E7E7E7",
    "text": "#222222",
}


def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "tiff"):
        out = FIG_DIR / f"{name}.{ext}"
        fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / name}.[pdf|svg|tiff]")
