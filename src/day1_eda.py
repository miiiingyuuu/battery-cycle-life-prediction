"""Day1 배터리 EDA: 데이터 읽기 → 피처 추출 → 통계 → 시각화.

원본 데이터는 수정하지 않으며, 모델 학습은 수행하지 않습니다.
전체 수명에서 얻는 knee·최종 용량은 설명용 정보로만 저장합니다.
"""

from pathlib import Path
import importlib.metadata
import json
import os
import re
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    os.environ.get(
        "BATTERY_DATA_DIR",
        (
            PROJECT_DIR / "data/battery-cycle"
            if (PROJECT_DIR / "data/battery-cycle").is_dir()
            else PROJECT_DIR.parent / "data/battery-cycle"
        ),
    )
)
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
BATCH_DATES = ["2017-05-12", "2018-02-20", "2018-04-12"]


def read_hdf5_array(mat_file, dataset):
    """MATLAB HDF5의 숫자 배열 또는 참조 배열을 1차원으로 읽습니다."""
    if h5py.check_dtype(ref=dataset.dtype) is not None:
        return np.concatenate(
            [
                np.asarray(mat_file[reference][()]).ravel()
                for reference in dataset[()].ravel()
            ]
        )
    return np.asarray(dataset[()]).ravel()


def fit_knee_candidate(cycle_numbers, capacity, window=21):
    """연속 구간 선형 모델을 단일 직선과 비교합니다. 설명용 분석입니다."""
    valid = (
        np.isfinite(capacity)
        & (capacity > 0)
        & (capacity < 1.5)
        & (cycle_numbers >= 10)
    )
    cycles = cycle_numbers[valid]
    values = capacity[valid]
    if len(cycles) < 100:
        return {
            key: np.nan
            for key in ["knee", "slope_before", "slope_after", "sse_gain", "bic_gain"]
        }
    smoothed = (
        pd.Series(values)
        .rolling(window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    # c + a*x + b*max(0, x-k): 분할점에서 용량이 연속인 모델입니다.
    linear_design = np.column_stack([np.ones(len(cycles)), cycles])
    linear_fit = np.linalg.lstsq(linear_design, smoothed, rcond=None)[0]
    linear_sse = float(np.sum((smoothed - linear_design @ linear_fit) ** 2))
    best = None
    for index in np.linspace(30, len(cycles) - 30, 60).astype(int):
        candidate = cycles[index]
        design = np.column_stack([linear_design, np.maximum(0, cycles - candidate)])
        coefficients = np.linalg.lstsq(design, smoothed, rcond=None)[0]
        sse = float(np.sum((smoothed - design @ coefficients) ** 2))
        if best is None or sse < best[0]:
            best = (sse, candidate, coefficients[1], coefficients[1] + coefficients[2])
    segmented_sse, candidate, before, after = best
    # 단일 모델 2개, 구간 모델 4개(분할점 탐색 포함)의 파라미터를 반영합니다.
    bic_gain = len(cycles) * np.log(
        max(linear_sse, 1e-16) / max(segmented_sse, 1e-16)
    ) - 2 * np.log(len(cycles))
    return {
        "knee": float(candidate),
        "slope_before": float(before),
        "slope_after": float(after),
        "sse_gain": 1 - segmented_sse / linear_sse if linear_sse > 1e-16 else 0.0,
        "bic_gain": float(bic_gain),
    }


def knee_diagnostics(cycle_numbers, capacity):
    """11·21·31회 평활 민감도와 사전 정의한 후보 지지 기준을 계산합니다."""
    fits = {
        window: fit_knee_candidate(cycle_numbers, capacity, window)
        for window in [11, 21, 31]
    }
    base = fits[21]
    spread = max(fit["knee"] for fit in fits.values()) - min(
        fit["knee"] for fit in fits.values()
    )
    relative_spread = spread / (cycle_numbers[-1] - cycle_numbers[0])
    return {
        **base,
        "knee_window11": fits[11]["knee"],
        "knee_window31": fits[31]["knee"],
        "knee_spread_cycles": spread,
        "knee_relative_spread": relative_spread,
        "knee_supported": bool(
            base["sse_gain"] >= 0.2
            and base["bic_gain"] > 10
            and base["slope_after"] < base["slope_before"] < 0
            and relative_spread <= 0.05
        ),
    }


def extract_cell(mat_file, batch_group, batch_number, cell_index):
    """한 셀의 초기 피처와 전체 수명 품질 정보를 추출합니다. 10·100회는 배열 인덱스 9·99로 확인합니다."""
    delta_curve = None
    cell_id = f"b{batch_number}c{cell_index}"
    summary_group = mat_file[batch_group["summary"][cell_index, 0]]
    cycle_group = mat_file[batch_group["cycles"][cell_index, 0]]
    voltage = read_hdf5_array(mat_file, mat_file[batch_group["Vdlin"][cell_index, 0]])
    summary_values = {
        field: read_hdf5_array(mat_file, summary_group[field])
        for field in ["cycle", "QDischarge", "IR", "Tavg", "Tmax", "Tmin", "chargetime"]
    }
    cycle_numbers = summary_values["cycle"]
    capacity = summary_values["QDischarge"]
    cycle_life = float(
        read_hdf5_array(mat_file, mat_file[batch_group["cycle_life"][cell_index, 0]])[0]
    )
    policy = "".join(
        (
            chr(int(character_code))
            for character_code in read_hdf5_array(
                mat_file, mat_file[batch_group["policy_readable"][cell_index, 0]]
            )
        )
    )
    policy_match = re.match(r"^([\d.]+)C\(([\d.]+)%\)-([\d.]+)C", policy)
    policy_match = list(policy_match.groups()) if policy_match else []
    record = {
        "batch": batch_number,
        "cell_id": cell_id,
        "cycle_life": cycle_life,
        "policy": policy,
        "n_summary": len(cycle_numbers),
        "n_cycles": cycle_group["Qdlin"].size,
        "last_QD": capacity[-1],
    }
    record["C1"] = float(policy_match[0]) if len(policy_match) >= 3 else np.nan
    record["switch_pct"] = float(policy_match[1]) if len(policy_match) >= 3 else np.nan
    record["C2"] = float(policy_match[2]) if len(policy_match) >= 3 else np.nan
    # 초기 피처: 10~100회만 사용하며, 전체 수명 품질 정보와 구분합니다.
    early_mask = (cycle_numbers >= 10) & (cycle_numbers <= 100)
    valid_capacity = np.isfinite(capacity) & (capacity > 0) & (capacity < 1.5)
    early_capacity_mask = early_mask & valid_capacity
    record["QD_early_mean"] = (
        float(np.mean(capacity[early_capacity_mask]))
        if early_capacity_mask.any()
        else np.nan
    )
    record["QD_early_slope"] = (
        float(
            np.polyfit(
                cycle_numbers[early_capacity_mask], capacity[early_capacity_mask], 1
            )[0]
        )
        if early_capacity_mask.sum() > 2
        else np.nan
    )
    for key in ["IR", "Tavg", "Tmax", "chargetime"]:
        values = summary_values[key][early_mask]
        values = values[np.isfinite(values)]
        if key == "IR":
            values = values[values > 0]
        record[key + "_mean"] = float(np.mean(values)) if len(values) else np.nan
        if key == "chargetime":
            record["chargetime_median"] = (
                float(np.median(values)) if len(values) else np.nan
            )
            record["chargetime_max"] = float(np.max(values)) if len(values) else np.nan
            record["chargetime_gt30_n"] = int((values > 30).sum())
    # 아래 품질 플래그와 knee는 설명용이며 모델 입력에 포함하지 않습니다.
    record["bad_QD_rows"] = int((~valid_capacity).sum())
    record["summary_length_mismatch"] = any(
        (len(values) != len(cycle_numbers) for values in summary_values.values())
    )
    record.update(knee_diagnostics(cycle_numbers, capacity))
    record["endpoint_quality_flag"] = bool(capacity[-1] > 0.885)
    capacity_10_summary = capacity[cycle_numbers == 10]
    capacity_100_summary = capacity[cycle_numbers == 100]
    record["QD100_minus_QD10"] = (
        float(capacity_100_summary[0] - capacity_10_summary[0])
        if len(capacity_10_summary) == len(capacity_100_summary) == 1
        and 0 < capacity_10_summary[0] < 1.5
        and 0 < capacity_100_summary[0] < 1.5
        else np.nan
    )
    capacity_curve = (cycle_numbers, capacity)
    # 실제 사이클 번호를 확인한 뒤 10·100회의 방전 곡선을 비교합니다.
    record["cycle_axis_aligned"] = bool(
        len(cycle_numbers) > 99
        and cycle_numbers[9] == 10
        and (cycle_numbers[99] == 100)
    )
    record["dq_valid"] = False
    if record["cycle_axis_aligned"] and cycle_group["Qdlin"].size >= 100:
        capacity_10 = read_hdf5_array(mat_file, mat_file[cycle_group["Qdlin"][9, 0]])
        capacity_100 = read_hdf5_array(mat_file, mat_file[cycle_group["Qdlin"][99, 0]])
        if (
            capacity_10.shape == capacity_100.shape == voltage.shape
            and np.isfinite(capacity_10).all()
            and np.isfinite(capacity_100).all()
        ):
            delta_capacity = capacity_100 - capacity_10
            delta_variance = float(np.var(delta_capacity))
            record["dq_valid"] = True
            record["dq_capacity_flag"] = bool(
                max(capacity_10.max(), capacity_100.max()) > 1.5
                or min(capacity_10.min(), capacity_100.min()) < -0.1
            )
            record.update(
                dq_var=delta_variance,
                dq_logvar=np.log10(max(delta_variance, 1e-12)),
                dq_min=float(delta_capacity.min()),
                dq_mean=float(delta_capacity.mean()),
            )
            # 고전압 기준점의 상수 오프셋 제거는 분산을 변경하지 않아야 합니다.
            anchor = int(np.argmax(voltage))
            centered_delta = delta_capacity - delta_capacity[anchor]
            core_mask = (voltage >= 2.1) & (voltage <= 3.2)
            record["dq_anchor_offset"] = float(delta_capacity[anchor])
            record["dq_centered_logvar"] = float(
                np.log10(max(np.var(centered_delta), 1e-12))
            )
            record["dq_core_logvar"] = float(
                np.log10(max(np.var(delta_capacity[core_mask]), 1e-12))
            )
            delta_curve = (voltage, delta_capacity)
    # 양의 전류 샘플 평균입니다. 시간 가중 평균과는 다릅니다.
    if cycle_group["I"].size >= 10:
        values = read_hdf5_array(mat_file, mat_file[cycle_group["I"][9, 0]])
        values = values[np.isfinite(values) & (values > 0)]
        record["I10_positive_mean"] = float(values.mean()) if len(values) else np.nan
        record["I10_positive_max"] = float(values.max()) if len(values) else np.nan
    return record, capacity_curve, delta_curve


def load_batches():
    """지정된 세 배치를 읽고 셀별 피처, 곡선, 파일 감사 정보를 반환합니다."""
    records = []
    capacity_curves = {}
    delta_curves = {}
    batch_audits = []
    for batch_number, date in enumerate(BATCH_DATES, start=1):
        mat_path = next(DATA_DIR.glob(date + "*.mat"))
        with h5py.File(mat_path, "r") as mat_file:
            batch_group = mat_file["batch"]
            cell_count = batch_group["summary"].size
            for cell_index in range(cell_count):
                record, capacity_curve, delta_curve = extract_cell(
                    mat_file, batch_group, batch_number, cell_index
                )
                cell_id = record["cell_id"]
                records.append(record)
                capacity_curves[cell_id] = capacity_curve
                if delta_curve is not None:
                    delta_curves[cell_id] = delta_curve
            # 기존 감사 기준에 맞춰 마지막 셀의 스키마와 전압 축을 기록합니다.
            summary_group = mat_file[batch_group["summary"][cell_count - 1, 0]]
            cycle_group = mat_file[batch_group["cycles"][cell_count - 1, 0]]
            voltage = read_hdf5_array(
                mat_file, mat_file[batch_group["Vdlin"][cell_count - 1, 0]]
            )
            batch_audits.append(
                {
                    "batch": batch_number,
                    "file": mat_path.name,
                    "bytes": mat_path.stat().st_size,
                    "cells": cell_count,
                    "summary_keys": list(summary_group),
                    "cycle_keys": list(cycle_group),
                    "voltage_min": float(voltage.min()),
                    "voltage_max": float(voltage.max()),
                    "voltage_points": len(voltage),
                }
            )
    return pd.DataFrame(records), capacity_curves, delta_curves, batch_audits


def summarize_batches(cell_features):
    """배치별 수명 분포와 유효 라벨 수를 집계합니다."""
    stats = []
    for batch_number, batch_cells in cell_features.groupby("batch"):
        cycle_life = batch_cells.cycle_life
        stats.append(
            {
                "batch": batch_number,
                "n": len(batch_cells),
                "label_n": int(cycle_life.notna().sum()),
                "label_missing": int(cycle_life.isna().sum()),
                "min": cycle_life.min(),
                "q25": cycle_life.quantile(0.25),
                "median": cycle_life.median(),
                "mean": cycle_life.mean(),
                "q75": cycle_life.quantile(0.75),
                "max": cycle_life.max(),
                "short_lt500": int((cycle_life < 500).sum()),
                "long_gt1000": int((cycle_life > 1000).sum()),
                "below550": int((cycle_life < 550).sum()),
                "dq_valid": int(batch_cells.dq_valid.sum()),
                "last_above088": int((batch_cells.last_QD > 0.88).sum()),
                "bad_QD_rows": int(batch_cells.bad_QD_rows.sum()),
            }
        )
    return pd.DataFrame(stats)


def calculate_correlations(cell_features):
    """피처별 결측 쌍을 제외하고 Pearson·Spearman 상관계수와 표본 수를 계산합니다."""
    features = [
        "dq_logvar",
        "dq_min",
        "dq_mean",
        "QD_early_mean",
        "QD_early_slope",
        "IR_mean",
        "Tavg_mean",
        "Tmax_mean",
        "chargetime_mean",
        "chargetime_median",
        "C1",
        "C2",
        "I10_positive_mean",
        "QD100_minus_QD10",
        "dq_core_logvar",
        "dq_centered_logvar",
    ]
    correlations = []
    for batch_number, batch_cells in cell_features.groupby("batch"):
        for feature_name in features:
            valid_pairs = batch_cells[[feature_name, "cycle_life"]].dropna()
            if len(valid_pairs) > 3 and valid_pairs[feature_name].nunique() > 1:
                correlations.append(
                    {
                        "batch": batch_number,
                        "feature": feature_name,
                        "n": len(valid_pairs),
                        "pearson": valid_pairs.corr().iloc[0, 1],
                        "spearman": spearmanr(
                            valid_pairs[feature_name], valid_pairs.cycle_life
                        ).statistic,
                    }
                )
    return pd.DataFrame(correlations)


def save_figure(filename):
    """현재 그림의 여백을 정리하고 파일로 저장한 뒤 닫습니다."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close()


def plot_cycle_life(cell_features):
    """세 배치의 수명 히스토그램을 같은 구간으로 비교합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for axis, (batch_number, batch_cells) in zip(axes, cell_features.groupby("batch")):
        axis.hist(
            batch_cells.cycle_life,
            bins=np.r_[np.arange(150, 2300, 150), 2300],
            color=["#287c8e", "#a7763b", "#6c6cb2"][batch_number - 1],
            edgecolor="white",
        )
        axis.set(
            title=f"Batch {batch_number} (labeled n={batch_cells.cycle_life.notna().sum()})",
            xlabel="Cycle life",
            ylabel="Cells",
            xlim=(150, 2300),
        )
    save_figure("01_life.png")


def plot_degradation(cell_features, capacity_curves):
    """전체 용량 곡선을 비교하며 수명 라벨 결측 셀은 회색으로 표시합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for axis, (batch_number, batch_cells) in zip(axes, cell_features.groupby("batch")):
        for _, cell in batch_cells.iterrows():
            cycle_numbers, capacity = capacity_curves[cell.cell_id]
            valid_capacity = np.isfinite(capacity) & (capacity > 0) & (capacity < 1.5)
            axis.plot(
                cycle_numbers,
                np.where(valid_capacity, capacity, np.nan),
                alpha=0.45,
                lw=0.7,
                color=(
                    plt.cm.viridis(min(cell.cycle_life / 2300, 1))
                    if np.isfinite(cell.cycle_life)
                    else "#999999"
                ),
            )
        axis.axhline(0.88, color="tomato", ls="--")
        axis.set(
            title=f"Batch {batch_number}",
            xlabel="Cycle",
            ylabel="QD (Ah)",
            ylim=(0.65, 1.18),
            xlim=(0, 2500),
        )
    save_figure("02_degradation.png")


def plot_delta_curves(cell_features, delta_curves):
    """장단수명 그룹의 ΔQ 곡선과 전압별 중앙값을 비교합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for axis, (batch_number, batch_cells) in zip(axes, cell_features.groupby("batch")):
        for label, condition, color in [
            ("Short <500", batch_cells.cycle_life < 500, "#c45252"),
            ("Long >1000", batch_cells.cycle_life > 1000, "#287c8e"),
        ]:
            group_cells = batch_cells[condition & batch_cells.dq_valid]
            for cell_id in group_cells.cell_id:
                voltage, delta_capacity = delta_curves[cell_id]
                axis.plot(voltage, delta_capacity, color=color, alpha=0.22, lw=0.7)
            if len(group_cells):
                assert all(
                    (
                        np.allclose(
                            delta_curves[cell_id][0],
                            delta_curves[group_cells.cell_id.iloc[0]][0],
                        )
                        for cell_id in group_cells.cell_id
                    )
                )
                voltage, delta_capacity = delta_curves[group_cells.cell_id.iloc[0]]
                axis.plot(
                    voltage,
                    np.median(
                        [delta_curves[cell_id][1] for cell_id in group_cells.cell_id],
                        axis=0,
                    ),
                    color=color,
                    lw=2,
                    label=f"{label}, n={len(group_cells)}",
                )
        axis.set(
            title=f"Batch {batch_number}", xlabel="Voltage (V)", ylabel="Q100-Q10 (Ah)"
        )
        axis.legend(fontsize=7)
    save_figure("03_delta.png")


def plot_delta_signal(cell_features):
    """ΔQ 로그 분산과 수명의 관계를 시각화합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for axis, (batch_number, batch_cells) in zip(axes, cell_features.groupby("batch")):
        axis.scatter(batch_cells.dq_logvar, batch_cells.cycle_life, s=18, alpha=0.65)
        axis.set(
            title=f"Batch {batch_number}",
            xlabel="log10 Var(delta Q)",
            ylabel="Cycle life",
        )
    save_figure("04_delta_signal.png")


def plot_policy_life(cell_features):
    """배치별 수명 하위·상위 3개 정책을 이름·표본 수와 함께 표시합니다."""
    figure, axes = plt.subplots(3, 1, figsize=(10, 8.5))
    for axis, (batch_number, cells) in zip(axes, cell_features.groupby("batch")):
        policies = cells.groupby("policy").cycle_life.agg(["mean", "std", "count"])
        policies = policies[policies["count"] > 0].sort_values("mean")
        selected = pd.concat([policies.head(3), policies.tail(3)]).loc[
            lambda frame: ~frame.index.duplicated()
        ]
        positions = np.arange(len(selected))
        axis.errorbar(
            selected["mean"],
            positions,
            xerr=selected["std"].fillna(0),
            fmt="o",
            capsize=3,
            color="#287c8e",
        )
        labels = [
            f"{name.replace('-newstructure', '*')}  (n={int(row['count'])})"
            for name, row in selected.iterrows()
        ]
        axis.set_yticks(positions, labels, fontsize=9)
        axis.set(
            title=f"Batch {batch_number}: lowest / highest 3 policies",
            xlabel="Mean cycle life +/- 1 SD",
        )
        axis.invert_yaxis()
    save_figure("05_policy.png")


def plot_relative_delta(cell_features, delta_curves):
    """배치 내 하위·상위 사분위 셀을 비교합니다. 절대 장단수명 정의와 구분합니다."""
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    group_stats = []
    for axis, (batch_number, cells) in zip(axes, cell_features.groupby("batch")):
        labeled = cells.dropna(subset=["cycle_life"])
        lower, upper = labeled.cycle_life.quantile([0.25, 0.75])
        for name, mask, color in [
            ("Bottom quartile", labeled.cycle_life <= lower, "#c45252"),
            ("Top quartile", labeled.cycle_life >= upper, "#287c8e"),
        ]:
            group = labeled[mask & labeled.dq_valid]
            voltage = delta_curves[group.cell_id.iloc[0]][0]
            assert all(
                np.allclose(delta_curves[cell_id][0], voltage)
                for cell_id in group.cell_id
            )
            values = np.array([delta_curves[cell_id][1] for cell_id in group.cell_id])
            axis.fill_between(
                voltage,
                np.quantile(values, 0.25, axis=0),
                np.quantile(values, 0.75, axis=0),
                color=color,
                alpha=0.15,
            )
            axis.plot(
                voltage,
                np.median(values, axis=0),
                color=color,
                label=f"{name}, n={len(group)}",
            )
            group_stats.append(
                {
                    "batch": batch_number,
                    "group": name,
                    "n": len(group),
                    "cutoff": lower if name.startswith("Bottom") else upper,
                    "life_median": group.cycle_life.median(),
                    "dq_logvar_median": group.dq_logvar.median(),
                }
            )
        axis.set(
            title=f"Batch {batch_number}", xlabel="Voltage (V)", ylabel="Q100-Q10 (Ah)"
        )
        axis.legend(fontsize=7)
    pd.DataFrame(group_stats).to_csv(
        RESULTS_DIR / "relative_delta_groups.csv", index=False
    )
    save_figure("09_delta_relative.png")


def save_diagnostic_summaries(cell_features):
    """Knee·전압 기준점·미완료 후보의 민감도 결과를 별도 표로 저장합니다."""
    rows = []
    for batch_number, cells in cell_features.groupby("batch"):
        labeled = cells.dropna(subset=["cycle_life"])
        eligible = labeled[~labeled.endpoint_quality_flag]
        rows.append(
            {
                "batch": batch_number,
                "n": len(cells),
                "knee_fit_n": int(cells.knee.notna().sum()),
                "knee_supported": int(cells.knee_supported.sum()),
                "sse_gain_median": cells.sse_gain.median(),
                "bic_gain_median": cells.bic_gain.median(),
                "knee_median": cells.knee.median(),
                "slope_before_median": cells.slope_before.median(),
                "slope_after_median": cells.slope_after.median(),
                "spread_median": cells.knee_spread_cycles.median(),
                "anchor_max_abs": cells.dq_anchor_offset.abs().max(),
                "centered_max_difference": (cells.dq_logvar - cells.dq_centered_logvar)
                .abs()
                .max(),
                "full_core_spearman": cells[["dq_logvar", "dq_core_logvar"]]
                .corr(method="spearman")
                .iloc[0, 1],
                "raw_label_n": len(labeled),
                "unflagged_label_n": len(eligible),
                "raw_dq_r": labeled.dq_logvar.corr(labeled.cycle_life),
                "unflagged_dq_r": eligible.dq_logvar.corr(eligible.cycle_life),
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "diagnostic_summary.csv", index=False)


def plot_charge_rate(cell_features):
    """첫 단계 충전 속도와 수명의 관계를 표시합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for axis, (batch_number, batch_cells) in zip(axes, cell_features.groupby("batch")):
        axis.scatter(
            batch_cells.C1,
            batch_cells.cycle_life,
            c=batch_cells.Tavg_mean,
            cmap="plasma",
            s=20,
        )
        axis.set(
            title=f"Batch {batch_number}",
            xlabel="First-stage C-rate",
            ylabel="Cycle life",
        )
    save_figure("06_crate.png")


def plot_feature_correlations(cell_features):
    """배치별 초기 피처 상관행렬을 표시합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    selected_features = [
        "dq_logvar",
        "QD_early_slope",
        "Tavg_mean",
        "Tmax_mean",
        "chargetime_mean",
        "IR_mean",
        "cycle_life",
    ]
    for axis, (batch_number, batch_cells) in zip(axes, cell_features.groupby("batch")):
        correlation_matrix = batch_cells[selected_features].corr()
        heatmap = axis.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_xticks(
            range(len(selected_features)), selected_features, rotation=90, fontsize=7
        )
        axis.set_yticks(range(len(selected_features)), selected_features, fontsize=7)
        axis.set_title(f"Batch {batch_number}")
        for row_index in range(len(selected_features)):
            for column_index in range(len(selected_features)):
                axis.text(
                    column_index,
                    row_index,
                    f"{correlation_matrix.iloc[row_index, column_index]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
    save_figure("07_corr.png")


def plot_knee_examples(cell_features, capacity_curves):
    """각 배치 첫 셀의 knee 후보를 표시합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for axis, (batch_number, batch_cells) in zip(axes, cell_features.groupby("batch")):
        cell = batch_cells.iloc[0]
        cycle_numbers, capacity = capacity_curves[cell.cell_id]
        valid_capacity = np.isfinite(capacity) & (capacity > 0) & (capacity < 1.5)
        axis.plot(cycle_numbers[valid_capacity], capacity[valid_capacity], lw=1)
        axis.axvline(cell.knee, color="tomato", ls="--")
        axis.set(
            title=(
                f"{cell.cell_id}: candidate {cell.knee:.0f}\n"
                f"{'Supported' if cell.knee_supported else 'Not supported'} "
                f"(SSE gain {cell.sse_gain:.1%})"
            ),
            xlabel="Cycle",
            ylabel="QD (Ah)",
        )
    save_figure("08_knee.png")


def save_results(cell_features, batch_stats, correlations, batch_audits):
    """표와 실행 환경을 저장합니다. 기존 CSV 컬럼 이름을 그대로 유지합니다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cell_features.to_csv(RESULTS_DIR / "cell_features.csv", index=False)
    batch_stats.to_csv(RESULTS_DIR / "batch_stats.csv", index=False)
    correlations.to_csv(RESULTS_DIR / "correlations.csv", index=False)
    cell_features.groupby(["batch", "policy"]).cycle_life.agg(
        ["count", "mean", "std"]
    ).to_csv(RESULTS_DIR / "policy_stats.csv")
    # EDA 필수 패키지의 누락은 숨기지 않습니다. PDF 생성용 패키지는
    # EDA 실행과 독립적이므로 설치되지 않았으면 감사 기록에만 남깁니다.
    versions = {
        package: importlib.metadata.version(package)
        for package in ["h5py", "numpy", "pandas", "matplotlib", "scipy"]
    }
    missing_optional_packages = []
    for package in ["reportlab", "pymupdf"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            missing_optional_packages.append(package)

    metadata = {
        "audits": batch_audits,
        "stats": batch_stats.to_dict(orient="records"),
        "python": sys.version,
        "versions": versions,
        "missing_optional_pdf_packages": missing_optional_packages,
        "seed": 42,
    }
    (RESULTS_DIR / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )


def main():
    """데이터 추출, 통계 저장, 아홉 EDA 그림 생성을 순서대로 실행합니다."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    cell_features, capacity_curves, delta_curves, batch_audits = load_batches()
    batch_stats = summarize_batches(cell_features)
    correlations = calculate_correlations(cell_features)
    save_results(cell_features, batch_stats, correlations, batch_audits)
    plot_cycle_life(cell_features)
    plot_degradation(cell_features, capacity_curves)
    plot_delta_curves(cell_features, delta_curves)
    plot_delta_signal(cell_features)
    plot_relative_delta(cell_features, delta_curves)
    save_diagnostic_summaries(cell_features)
    plot_policy_life(cell_features)
    plot_charge_rate(cell_features)
    plot_feature_correlations(cell_features)
    plot_knee_examples(cell_features, capacity_curves)
    print(batch_stats.to_string(index=False))
    print("\nCORRELATIONS\n", correlations.round(3).to_string(index=False))
    print(
        "\nQC\n",
        cell_features[
            ["cell_id", "cycle_life", "n_summary", "last_QD", "bad_QD_rows", "dq_valid"]
        ].to_string(index=False),
    )


if __name__ == "__main__":
    main()
