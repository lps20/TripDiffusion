import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_SOURCES: List[Tuple[str, str]] = [
    ("Real World", "data/test_data.csv"),
    ("VAE", "exp/baseline/VAE_gene.csv"),
    ("CTGAN", "exp/baseline/CTGAN_gene.csv"),
    ("DATGAN", "exp/baseline/DATGAN_gene.csv"),
    ("DDPM+TF", "exp/baseline/DDPM_TF_gene.csv"),
    ("D3PM+TF", "exp/tsf/generated_samples.csv"),
    ("D3PM+SC3T", "exp/hcd_v2/generated_samples.csv"),
]


def _model_panel_titles(model_names: Sequence[str]) -> List[str]:
    return [f"({chr(ord('a') + i)}) {name}" for i, name in enumerate(model_names)]

AGE_LABELS: List[str] = [
    "0-11",
    "12-17",
    "18-22",
    "23-27",
    "28-32",
    "33-37",
    "38-42",
    "43-47",
    "48-52",
    "53-57",
    "58-62",
    "63-69",
    "70+",
]

AGE_MAP: Dict[int, str] = {i: AGE_LABELS[i] for i in range(len(AGE_LABELS))}
SEX_MAP: Dict[int, str] = {0: "Male", 1: "Female"}

ACTIVITY_MAP: Dict[int, str] = {
    0: "Home",
    1: "Travel",
    2: "Work",
    3: "Education",
    4: "Academy",
    5: "Leisure",
    6: "Shopping",
    7: "Escort",
    8: "Other",
}
ACTIVITY_ORDER: List[str] = [
    "Home",
    "Travel",
    "Work",
    "Education",
    "Academy",
    "Leisure",
    "Shopping",
    "Escort",
    "Other",
]

MODE_MAP: Dict[int, str] = {
    0: "Start",
    1: "End",
    2: "Stay",
    3: "Walk",
    4: "Transit",
    5: "Car (driver)",
    6: "Car (passenger)",
    7: "Bicycle",
    8: "Taxi",
}
MODE_ORDER: List[str] = [
    "Bicycle",
    "Car (driver)",
    "Car (passenger)",
    "Taxi",
    "Transit",
    "Walk",
]


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 20,
            "axes.labelsize": 24,
            "axes.titlesize": 22,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _compute_joint_pct(
    df: pd.DataFrame,
    row_col: str,
    row_map: Dict[int, str],
    row_order: Sequence[str],
    col_col: str,
    col_map: Dict[int, str],
    col_order: Sequence[str],
    mask: Optional[pd.Series] = None,
) -> np.ndarray:
    required = [row_col, col_col]
    if mask is not None:
        for c in mask.index.names if hasattr(mask.index, "names") else []:
            if c is not None and c not in required:
                required.append(c)
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Column `{c}` not found in dataframe.")

    work = df.copy()
    work[row_col] = pd.to_numeric(work[row_col], errors="coerce")
    work[col_col] = pd.to_numeric(work[col_col], errors="coerce")
    work = work.dropna(subset=[row_col, col_col]).copy()
    work[row_col] = work[row_col].astype(int)
    work[col_col] = work[col_col].astype(int)

    if mask is not None:
        mask_local = mask.reindex(work.index, fill_value=False)
        work = work.loc[mask_local].copy()

    work["_row"] = work[row_col].map(row_map)
    work["_col"] = work[col_col].map(col_map)
    work = work.dropna(subset=["_row", "_col"])
    work = work[work["_row"].isin(row_order) & work["_col"].isin(col_order)]

    counts = pd.crosstab(work["_row"], work["_col"])
    counts = counts.reindex(index=row_order, columns=col_order, fill_value=0)
    total = counts.to_numpy().sum()
    if total <= 0:
        return np.zeros((len(row_order), len(col_order)), dtype=float)
    return counts.to_numpy(dtype=float) / float(total) * 100.0


def _joint_grid_shape(n_models: int) -> Tuple[int, int]:
    """Match plot_marginal_distributions joint-style figures: one row if <=4 else two rows."""
    n_rows = 1 if n_models <= 4 else 2
    n_cols = int(np.ceil(n_models / n_rows))
    return n_rows, n_cols


def _plot_joint_heatmap_comparison(
    mats: Sequence[np.ndarray],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    subplot_titles: Sequence[str],
    x_label: str,
    y_label: str,
    out_png: str,
    out_pdf: str,
    *,
    layout: str,
    fig_size: Tuple[float, float],
    cell_fontsize: int = 18,
    tick_fontsize: int = 18,
) -> None:
    n_models = len(mats)
    if layout == "stacked":
        nrows, ncols = n_models, 1
    elif layout == "grid":
        nrows, ncols = _joint_grid_shape(n_models)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    if layout == "grid":
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=fig_size,
            gridspec_kw={"wspace": 0.14, "hspace": 0.22},
            constrained_layout=True,
        )
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, constrained_layout=True)
    axes_flat = np.atleast_2d(axes).ravel()

    vmax = max(float(np.max(m)) for m in mats)
    if vmax <= 0:
        vmax = 1.0

    image_ref = None
    for i, ax in enumerate(axes_flat):
        if i >= n_models:
            ax.set_visible(False)
            continue

        mat = mats[i]
        image_ref = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(subplot_titles[i], pad=6)

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))

        if layout == "stacked":
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xticklabels(col_labels, fontsize=tick_fontsize, rotation=0)
            ax.set_yticklabels(row_labels, fontsize=tick_fontsize)
        else:
            if i // ncols == nrows - 1:
                ax.set_xticklabels(col_labels, fontsize=tick_fontsize, rotation=0)
            else:
                ax.set_xticklabels([])
            if i % ncols == 0:
                ax.set_yticklabels(row_labels, fontsize=tick_fontsize)
            else:
                ax.set_yticklabels([])

        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                color = "white" if val > 0.58 * vmax else "#1f1f1f"
                ax.text(c, r, f"{val:.2f}%", ha="center", va="center", fontsize=cell_fontsize, color=color)

        for spine in ax.spines.values():
            spine.set_visible(False)

    if layout == "grid":
        fig.supxlabel(x_label, y=-0.02, fontsize=plt.rcParams["axes.labelsize"])
        fig.supylabel(y_label, x=-0.02, fontsize=plt.rcParams["axes.labelsize"])
    used_axes = [axes_flat[k] for k in range(n_models)]
    cbar = fig.colorbar(image_ref, ax=used_axes, fraction=0.022, pad=0.01)
    cbar.set_label("Percentage of total trips")
    cbar.ax.tick_params(labelsize=20)

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _configure_style()
    os.makedirs("figs", exist_ok=True)

    model_names = [name for name, _ in MODEL_SOURCES]
    subplot_titles = _model_panel_titles(model_names)
    n_models = len(MODEL_SOURCES)

    loaded: List[Tuple[str, pd.DataFrame]] = [(name, _load_csv(path)) for name, path in MODEL_SOURCES]

    age_act_mats: List[np.ndarray] = []
    age_mode_mats: List[np.ndarray] = []
    gender_act_mats: List[np.ndarray] = []
    gender_mode_mats: List[np.ndarray] = []

    for _, df in loaded:
        mask_travel_mode = (pd.to_numeric(df["act_num"], errors="coerce") == 1) & (
            pd.to_numeric(df["mode_num"], errors="coerce").isin([3, 4, 5, 6, 7, 8])
        )

        age_act_mats.append(
            _compute_joint_pct(
                df=df,
                row_col="act_num",
                row_map=ACTIVITY_MAP,
                row_order=ACTIVITY_ORDER,
                col_col="age_code",
                col_map=AGE_MAP,
                col_order=AGE_LABELS,
            )
        )

        age_mode_mats.append(
            _compute_joint_pct(
                df=df,
                row_col="mode_num",
                row_map=MODE_MAP,
                row_order=MODE_ORDER,
                col_col="age_code",
                col_map=AGE_MAP,
                col_order=AGE_LABELS,
                mask=mask_travel_mode,
            )
        )

        gender_act_mats.append(
            _compute_joint_pct(
                df=df,
                row_col="act_num",
                row_map=ACTIVITY_MAP,
                row_order=ACTIVITY_ORDER,
                col_col="sex",
                col_map=SEX_MAP,
                col_order=["Male", "Female"],
            )
        )

        gender_mode_mats.append(
            _compute_joint_pct(
                df=df,
                row_col="mode_num",
                row_map=MODE_MAP,
                row_order=MODE_ORDER,
                col_col="sex",
                col_map=SEX_MAP,
                col_order=["Male", "Female"],
                mask=mask_travel_mode,
            )
        )

    _plot_joint_heatmap_comparison(
        mats=age_act_mats,
        row_labels=ACTIVITY_ORDER,
        col_labels=AGE_LABELS,
        subplot_titles=subplot_titles,
        x_label="Age Group",
        y_label="Activity",
        out_png="figs/age_activity_joint_vertical.png",
        out_pdf="figs/age_activity_joint_vertical.pdf",
        layout="stacked",
        fig_size=(13.6, max(4.35 * n_models, 6.0)),
        cell_fontsize=18,
    )

    _plot_joint_heatmap_comparison(
        mats=age_mode_mats,
        row_labels=MODE_ORDER,
        col_labels=AGE_LABELS,
        subplot_titles=subplot_titles,
        x_label="Age Group",
        y_label="Mode",
        out_png="figs/age_mode_joint_vertical.png",
        out_pdf="figs/age_mode_joint_vertical.pdf",
        layout="stacked",
        fig_size=(13.6, max(4.0 * n_models, 5.5)),
        cell_fontsize=18,
    )

    g_rows, g_cols = _joint_grid_shape(n_models)
    gender_fig_w = 4.15 * g_cols + 2.35
    gender_fig_h = 5.35 * g_rows + 0.55

    _plot_joint_heatmap_comparison(
        mats=gender_act_mats,
        row_labels=ACTIVITY_ORDER,
        col_labels=["Male", "Female"],
        subplot_titles=subplot_titles,
        x_label="Gender",
        y_label="Activity",
        out_png="figs/gender_activity_joint_vertical.png",
        out_pdf="figs/gender_activity_joint_vertical.pdf",
        layout="grid",
        fig_size=(gender_fig_w, gender_fig_h),
        cell_fontsize=16,
        tick_fontsize=17,
    )

    _plot_joint_heatmap_comparison(
        mats=gender_mode_mats,
        row_labels=MODE_ORDER,
        col_labels=["Male", "Female"],
        subplot_titles=subplot_titles,
        x_label="Gender",
        y_label="Mode",
        out_png="figs/gender_mode_joint_vertical.png",
        out_pdf="figs/gender_mode_joint_vertical.pdf",
        layout="grid",
        fig_size=(gender_fig_w, gender_fig_h * 0.9),
        cell_fontsize=16,
        tick_fontsize=17,
    )


if __name__ == "__main__":
    main()
