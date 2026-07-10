import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import os
from typing import Dict, List, Sequence, Tuple

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
    "Academy",
    "Education",
    "Escort",
    "Home",
    "Leisure",
    "Other",
    "Shopping",
    "Travel",
    "Work",
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
    "Walk",
    "Transit",
    "Car (driver)",
    "Car (passenger)",
    "Bicycle",
    "Taxi",
]

ACTIVITY_TIME_ORDER: List[str] = [
    "Academy",
    "Education",
    "Escort",
    "Home",
    "Leisure",
    "Other",
    "Shopping",
    "Work",
]

TIME_SLOTS: np.ndarray = np.arange(0, 241)
OD_TOP_K: int = 12


def _configure_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 17,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _to_int_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Column `{col}` not found in dataframe columns: {list(df.columns)}")
    return pd.to_numeric(df[col], errors="coerce").dropna().astype(int)


def _activity_percentages(df: pd.DataFrame) -> np.ndarray:
    act = _to_int_series(df, "act_num").map(ACTIVITY_MAP).dropna()
    counts = act.value_counts().reindex(ACTIVITY_ORDER, fill_value=0)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(ACTIVITY_ORDER), dtype=float)
    return (counts / total * 100.0).to_numpy()


def _mode_percentages(df: pd.DataFrame) -> np.ndarray:
    act = _to_int_series(df, "act_num")
    mode = _to_int_series(df, "mode_num")
    aligned = pd.DataFrame({"act_num": act, "mode_num": mode}).dropna()
    travel_modes = aligned.loc[aligned["act_num"] == 1, "mode_num"]
    mode_labels = travel_modes.map(MODE_MAP).dropna()
    mode_labels = mode_labels[mode_labels.isin(MODE_ORDER)]
    counts = mode_labels.value_counts().reindex(MODE_ORDER, fill_value=0)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(MODE_ORDER), dtype=float)
    return (counts / total * 100.0).to_numpy()


def _prepare_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["act_num", "mode_num", "start_time_num_6", "trip_time_num_6"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for joint-time plot: {missing}")

    out = df[required].copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    for c in required:
        out[c] = out[c].astype(int)

    out["start_time_num_6"] = out["start_time_num_6"].clip(lower=0, upper=240)
    out["trip_time_num_6"] = out["trip_time_num_6"].clip(lower=0, upper=240)
    return out


def _prepare_od_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["start_zcode_num", "end_zcode_num"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for OD matrix: {missing}")
    out = df[required].copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    for c in required:
        out[c] = out[c].astype(int)
    return out


def _od_matrix_row_percentage(df: pd.DataFrame) -> pd.DataFrame:
    od = _prepare_od_columns(df)
    trans = pd.crosstab(od["start_zcode_num"], od["end_zcode_num"])
    row_sum = trans.sum(axis=1)
    pct = trans.divide(row_sum.where(row_sum > 0, np.nan), axis=0) * 100.0
    return pct.fillna(0.0)


def _select_top_od_axes(df_real: pd.DataFrame, top_k: int) -> Tuple[List[int], List[int]]:
    od_real = _prepare_od_columns(df_real)
    trans_real = pd.crosstab(od_real["start_zcode_num"], od_real["end_zcode_num"])
    top_starts = trans_real.sum(axis=1).nlargest(top_k).index.astype(int).tolist()
    top_ends = trans_real.sum(axis=0).nlargest(top_k).index.astype(int).tolist()
    return top_starts, top_ends


def _subset_od_matrix(pct: pd.DataFrame, start_order: Sequence[int], end_order: Sequence[int]) -> np.ndarray:
    pct_sub = pct.reindex(index=start_order, columns=end_order, fill_value=0.0)
    return pct_sub.to_numpy(dtype=float)


def _expand_records(labels: Sequence[str], starts: Sequence[int], durations: Sequence[int]) -> pd.DataFrame:
    records: List[Tuple[str, int]] = []
    for label, start, duration in zip(labels, starts, durations):
        end = min(int(start) + int(duration), 240)
        for t in range(int(start), end):
            records.append((label, t))
    if not records:
        return pd.DataFrame(columns=["label", "time"])
    return pd.DataFrame(records, columns=["label", "time"])


def _column_normalized_matrix(counts: pd.DataFrame, row_order: Sequence[str]) -> np.ndarray:
    counts = counts.reindex(index=row_order, fill_value=0)
    counts = counts.reindex(columns=TIME_SLOTS, fill_value=0)
    col_sums = counts.sum(axis=0)
    pct = counts.divide(col_sums.where(col_sums > 0, np.nan), axis=1) * 100.0
    pct = pct.fillna(0.0)
    return pct.to_numpy()


def _activity_time_joint_percentages(df: pd.DataFrame) -> np.ndarray:
    td = _prepare_time_columns(df)
    labels = td["act_num"].map(ACTIVITY_MAP)
    valid = labels.isin(ACTIVITY_TIME_ORDER)
    expanded = _expand_records(
        labels[valid].tolist(),
        td.loc[valid, "start_time_num_6"].tolist(),
        td.loc[valid, "trip_time_num_6"].tolist(),
    )
    if expanded.empty:
        return np.zeros((len(ACTIVITY_TIME_ORDER), len(TIME_SLOTS)), dtype=float)
    counts = expanded.groupby(["label", "time"]).size().unstack(fill_value=0)
    return _column_normalized_matrix(counts=counts, row_order=ACTIVITY_TIME_ORDER)


def _mode_time_joint_percentages(df: pd.DataFrame) -> np.ndarray:
    td = _prepare_time_columns(df)
    mode_labels = td["mode_num"].map(MODE_MAP)
    valid = (td["act_num"] == 1) & mode_labels.isin(MODE_ORDER)
    expanded = _expand_records(
        mode_labels[valid].tolist(),
        td.loc[valid, "start_time_num_6"].tolist(),
        td.loc[valid, "trip_time_num_6"].tolist(),
    )
    if expanded.empty:
        return np.zeros((len(MODE_ORDER), len(TIME_SLOTS)), dtype=float)
    counts = expanded.groupby(["label", "time"]).size().unstack(fill_value=0)
    return _column_normalized_matrix(counts=counts, row_order=MODE_ORDER)


def _plot_vertical_heatmap_comparison(
    values_by_model: Sequence[np.ndarray],
    row_labels: Sequence[str],
    model_labels: Sequence[str],
    y_label: str,
    out_png: str,
    out_pdf: str,
) -> None:
    n_models = len(model_labels)
    n_rows = 1
    n_cols = n_models
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.95 * n_cols + 1.0, 4.0),
        gridspec_kw={"wspace": 0.16, "hspace": 0.0},
        constrained_layout=True,
    )
    axes_flat = np.atleast_2d(axes).ravel()

    cmap = plt.get_cmap("YlGnBu")
    image_ref = None

    for i, ax in enumerate(axes_flat):
        if i >= n_models:
            ax.set_visible(False)
            continue

        col = np.asarray(values_by_model[i], dtype=float).reshape(-1, 1)
        image_ref = ax.imshow(col, cmap=cmap, vmin=0.0, vmax=100.0, aspect="auto")

        for r, pct in enumerate(col[:, 0]):
            text_color = "white" if pct >= 55 else "#1f1f1f"
            ax.text(
                0,
                r,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=17,
                color=text_color,
            )

        ax.set_xticks([])
        ax.set_yticks(np.arange(len(row_labels)))
        if i % n_cols == 0:
            ax.set_yticklabels(row_labels, fontsize=22)
        else:
            ax.set_yticklabels([])
        ax.set_title(model_labels[i], pad=6)

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.supylabel(y_label, x=-0.01)
    used_axes = [axes_flat[k] for k in range(n_models)]
    cbar = fig.colorbar(image_ref, ax=used_axes, fraction=0.02, pad=0.01)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_joint_time_heatmap_comparison(
    values_by_model: Sequence[np.ndarray],
    row_labels: Sequence[str],
    model_labels: Sequence[str],
    y_label: str,
    out_png: str,
    out_pdf: str,
) -> None:
    n_models = len(model_labels)
    n_rows = 1 if n_models <= 4 else 2
    n_cols = int(np.ceil(n_models / n_rows))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.3 * n_cols + 1.1, 2.9 * n_rows + 0.35),
        gridspec_kw={"wspace": 0.14, "hspace": 0.18},
        constrained_layout=True,
    )
    axes_flat = np.atleast_2d(axes).ravel()

    cmap = plt.get_cmap("YlGnBu")
    image_ref = None
    tick_pos = np.arange(0, 241, 80)
    tick_labels = [f"{t / 10:.1f}h" for t in tick_pos]

    for i, ax in enumerate(axes_flat):
        if i >= n_models:
            ax.set_visible(False)
            continue

        mat = np.asarray(values_by_model[i], dtype=float)
        image_ref = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=100.0, aspect="auto", interpolation="nearest")

        ax.set_xticks(tick_pos)
        if i // n_cols == n_rows - 1:
            ax.set_xticklabels(tick_labels, fontsize=17)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="x", labelrotation=0)

        ax.set_yticks(np.arange(len(row_labels)))
        if i % n_cols == 0:
            ax.set_yticklabels(row_labels, fontsize=22)
        else:
            ax.set_yticklabels([])
        ax.set_title(model_labels[i], pad=6)

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.supxlabel("Time", y=-0.04)
    fig.supylabel(y_label, x=-0.02)
    used_axes = [axes_flat[k] for k in range(n_models)]
    cbar = fig.colorbar(image_ref, ax=used_axes, fraction=0.02, pad=0.01)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_od_matrix_heatmap_comparison(
    values_by_model: Sequence[np.ndarray],
    model_labels: Sequence[str],
    start_order: Sequence[int],
    end_order: Sequence[int],
    out_png: str,
    out_pdf: str,
) -> None:
    n_models = len(model_labels)
    n_rows = 1 if n_models <= 4 else 2
    n_cols = int(np.ceil(n_models / n_rows))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols + 1.4, 3.0 * n_rows + 0.5),
        gridspec_kw={"wspace": 0.16, "hspace": 0.2},
        constrained_layout=True,
    )
    axes_flat = np.atleast_2d(axes).ravel()
    cmap = plt.get_cmap("YlGnBu")
    image_ref = None

    for i, ax in enumerate(axes_flat):
        if i >= n_models:
            ax.set_visible(False)
            continue
        mat = np.asarray(values_by_model[i], dtype=float)
        image_ref = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=100.0, aspect="auto", interpolation="nearest")

        x_idx = np.arange(len(end_order))
        y_idx = np.arange(len(start_order))
        x_step = 1 if len(end_order) <= 8 else 2
        y_step = 1 if len(start_order) <= 8 else 2
        ax.set_xticks(x_idx[::x_step])
        ax.set_yticks(y_idx[::y_step])

        if i // n_cols == n_rows - 1:
            ax.set_xticklabels([str(end_order[j]) for j in x_idx[::x_step]], fontsize=17)
        else:
            ax.set_xticklabels([])

        if i % n_cols == 0:
            ax.set_yticklabels([str(start_order[j]) for j in y_idx[::y_step]], fontsize=17)
        else:
            ax.set_yticklabels([])

        ax.set_title(model_labels[i], pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.supxlabel("End Zone Code", y=-0.02)
    fig.supylabel("Start Zone Code", x=-0.02)
    used_axes = [axes_flat[k] for k in range(n_models)]
    cbar = fig.colorbar(image_ref, ax=used_axes, fraction=0.02, pad=0.01)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _configure_academic_style()
    os.makedirs("figs", exist_ok=True)

    model_labels: List[str] = []
    activity_values: List[np.ndarray] = []
    mode_values: List[np.ndarray] = []
    activity_time_values: List[np.ndarray] = []
    mode_time_values: List[np.ndarray] = []
    od_values: List[np.ndarray] = []
    od_start_order: List[int] = []
    od_end_order: List[int] = []

    for idx, (model_name, csv_path) in enumerate(MODEL_SOURCES):
        df = _load_csv(csv_path)
        if idx == 0:
            od_start_order, od_end_order = _select_top_od_axes(df_real=df, top_k=OD_TOP_K)
        model_labels.append(model_name)
        activity_values.append(_activity_percentages(df))
        mode_values.append(_mode_percentages(df))
        activity_time_values.append(_activity_time_joint_percentages(df))
        mode_time_values.append(_mode_time_joint_percentages(df))
        od_values.append(_subset_od_matrix(_od_matrix_row_percentage(df), od_start_order, od_end_order))

    _plot_vertical_heatmap_comparison(
        values_by_model=activity_values,
        row_labels=ACTIVITY_ORDER,
        model_labels=model_labels,
        y_label="Activity",
        out_png="figs/activity_marginal_comparison.png",
        out_pdf="figs/activity_marginal_comparison.pdf",
    )

    _plot_vertical_heatmap_comparison(
        values_by_model=mode_values,
        row_labels=MODE_ORDER,
        model_labels=model_labels,
        y_label="Mode",
        out_png="figs/mode_marginal_comparison.png",
        out_pdf="figs/mode_marginal_comparison.pdf",
    )

    _plot_joint_time_heatmap_comparison(
        values_by_model=activity_time_values,
        row_labels=ACTIVITY_TIME_ORDER,
        model_labels=model_labels,
        y_label="Activity Type",
        out_png="figs/activity_time_joint_comparison.png",
        out_pdf="figs/activity_time_joint_comparison.pdf",
    )

    _plot_joint_time_heatmap_comparison(
        values_by_model=mode_time_values,
        row_labels=MODE_ORDER,
        model_labels=model_labels,
        y_label="Mode Type",
        out_png="figs/mode_time_joint_comparison.png",
        out_pdf="figs/mode_time_joint_comparison.pdf",
    )

    _plot_od_matrix_heatmap_comparison(
        values_by_model=od_values,
        model_labels=model_labels,
        start_order=od_start_order,
        end_order=od_end_order,
        out_png="figs/od_matrix_comparison.png",
        out_pdf="figs/od_matrix_comparison.pdf",
    )


if __name__ == "__main__":
    main()
