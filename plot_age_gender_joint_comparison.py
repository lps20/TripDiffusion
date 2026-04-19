import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_SOURCES: List[Tuple[str, str]] = [
    ("Real World", "data/test_data.csv"),
    ("DDPM+TF", "exp/baseline/DDPM_TF_gene.csv"),
    ("D3PM+SC3T", "exp/hcd_v2/generated_samples.csv"),
]

MODEL_SUBTITLES: List[str] = [
    "(a) Real World",
    "(b) DDPM+TF",
    "(c) D3PM+SC3T",
]

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
            "font.size": 18,
            "axes.labelsize": 22,
            "axes.titlesize": 20,
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


def _plot_vertical_comparison(
    mats: Sequence[np.ndarray],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    subplot_titles: Sequence[str],
    x_label: str,
    y_label: str,
    out_png: str,
    out_pdf: str,
    fig_size: Tuple[float, float],
    nrows: int = 3,
    ncols: int = 1,
) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, constrained_layout=True)
    axes_flat = np.array(axes).reshape(-1)
    vmax = max(float(np.max(m)) for m in mats)
    if vmax <= 0:
        vmax = 1.0

    image_ref = None
    for i, ax in enumerate(axes_flat):
        mat = mats[i]
        image_ref = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(subplot_titles[i], pad=6)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=16, rotation=0)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=16)

        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                color = "white" if val > 0.58 * vmax else "#1f1f1f"
                ax.text(c, r, f"{val:.2f}%", ha="center", va="center", fontsize=15, color=color)

        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(image_ref, ax=axes_flat, fraction=0.022, pad=0.01)
    cbar.set_label("Percentage of total trips")
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _configure_style()
    os.makedirs("figs", exist_ok=True)

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

    _plot_vertical_comparison(
        mats=age_act_mats,
        row_labels=ACTIVITY_ORDER,
        col_labels=AGE_LABELS,
        subplot_titles=MODEL_SUBTITLES,
        x_label="Age Group",
        y_label="Activity",
        out_png="figs/age_activity_joint_vertical.png",
        out_pdf="figs/age_activity_joint_vertical.pdf",
        fig_size=(13.6, 13.8),
        nrows=3,
        ncols=1,
    )

    _plot_vertical_comparison(
        mats=age_mode_mats,
        row_labels=MODE_ORDER,
        col_labels=AGE_LABELS,
        subplot_titles=MODEL_SUBTITLES,
        x_label="Age Group",
        y_label="Mode",
        out_png="figs/age_mode_joint_vertical.png",
        out_pdf="figs/age_mode_joint_vertical.pdf",
        fig_size=(13.6, 12.4),
        nrows=3,
        ncols=1,
    )

    _plot_vertical_comparison(
        mats=gender_act_mats,
        row_labels=ACTIVITY_ORDER,
        col_labels=["Male", "Female"],
        subplot_titles=MODEL_SUBTITLES,
        x_label="Gender",
        y_label="Activity",
        out_png="figs/gender_activity_joint_vertical.png",
        out_pdf="figs/gender_activity_joint_vertical.pdf",
        fig_size=(15.2, 5.8),
        nrows=1,
        ncols=3,
    )

    _plot_vertical_comparison(
        mats=gender_mode_mats,
        row_labels=MODE_ORDER,
        col_labels=["Male", "Female"],
        subplot_titles=MODEL_SUBTITLES,
        x_label="Gender",
        y_label="Mode",
        out_png="figs/gender_mode_joint_vertical.png",
        out_pdf="figs/gender_mode_joint_vertical.pdf",
        fig_size=(15.2, 5.2),
        nrows=1,
        ncols=3,
    )


if __name__ == "__main__":
    main()
