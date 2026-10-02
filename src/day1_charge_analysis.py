"""초기 충전 전류 패턴과 충전시간 기록을 원본에서 확인합니다."""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import day1_eda as eda


def load_charge_records():
    """각 배치 고정 예시 2셀의 10회 전류와 모든 셀의 초기 충전시간을 읽습니다."""
    patterns = {}
    durations = []
    for batch, date in enumerate(eda.BATCH_DATES, start=1):
        with h5py.File(next(eda.DATA_DIR.glob(date + "*.mat")), "r") as mat:
            group = mat["batch"]
            for index in range(group["summary"].size):
                cell_id = f"b{batch}c{index}"
                summary = mat[group["summary"][index, 0]]
                cycles = eda.read_hdf5_array(mat, summary["cycle"])
                times = eda.read_hdf5_array(mat, summary["chargetime"])
                for cycle, duration in zip(cycles, times):
                    if 10 <= cycle <= 100:
                        durations.append(
                            {
                                "batch": batch,
                                "cell_id": cell_id,
                                "cycle": cycle,
                                "chargetime": duration,
                            }
                        )
                # 라벨로 예시를 고르지 않고 사전에 정한 인덱스를 사용합니다.
                if index not in [0, 14]:
                    continue
                internal = mat[group["cycles"][index, 0]]
                time = eda.read_hdf5_array(mat, mat[internal["t"][9, 0]])
                current = eda.read_hdf5_array(mat, mat[internal["I"][9, 0]])
                patterns[cell_id] = (time, current)
    return patterns, pd.DataFrame(durations)


def plot_current_patterns(patterns, cells):
    """전체 10회 사이클의 양·음 전류를 보존해 충전 단계와 방전을 구분합니다."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharey=True)
    for axis, (cell_id, (time, current)) in zip(axes.ravel(), patterns.items()):
        row = cells.set_index("cell_id").loc[cell_id]
        valid = np.isfinite(time) & np.isfinite(current)
        axis.plot(time[valid], current[valid], lw=1)
        axis.axhline(0, color="gray", lw=0.6)
        # 명목 단계 전류로 실제 양의 전류 유지 구간과 전환을 대조합니다.
        axis.axhline(row.C1 * 1.1, color="orange", ls="--", lw=0.7, label="C1 × 1.1 Ah")
        axis.axhline(row.C2 * 1.1, color="green", ls=":", lw=0.7, label="C2 × 1.1 Ah")
        axis.set(
            title=f"{cell_id}: {row.policy}",
            xlabel="Cycle 10 time (min)",
            ylabel="I (A)",
        )
    axes[0, 0].legend(fontsize=7)
    eda.save_figure("10_current_patterns.png")


def plot_charge_time_records(durations):
    """평균을 왜곡하는 개별 기록을 숨기지 않고 로그 축으로 보여줍니다."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for axis, (batch, rows) in zip(axes, durations.groupby("batch")):
        for cell_id, values in rows.groupby("cell_id"):
            axis.plot(
                values.cycle,
                values.chargetime.where(values.chargetime > 0),
                lw=0.5,
                alpha=0.4,
            )
        axis.set(
            title=f"B{batch}: cycles 10–100", xlabel="Cycle", ylabel="chargetime (min)"
        )
        axis.set_yscale("log")
    eda.save_figure("11_charge_time.png")


def save_charge_audit(durations, cells):
    """30분은 기술적 집계 경계일 뿐 삭제·정상 판정 기준이 아닙니다."""
    rows = []
    for batch, values in durations.groupby("batch"):
        high = values[values.chargetime > 30]
        rows.append(
            {
                "batch": batch,
                "early_records_n": len(values),
                "gt30_records_n": len(high),
                "gt30_cells_n": high.cell_id.nunique(),
                "max_minutes": values.chargetime.max(),
            }
        )
    pd.DataFrame(rows).to_csv(eda.RESULTS_DIR / "charge_time_audit.csv", index=False)
    durations.to_csv(eda.RESULTS_DIR / "early_charge_records.csv", index=False)
    cells[
        [
            "cell_id",
            "batch",
            "policy",
            "chargetime_mean",
            "chargetime_median",
            "chargetime_max",
            "chargetime_gt30_n",
        ]
    ].to_csv(eda.RESULTS_DIR / "charge_time_cells.csv", index=False)


def plot_current_degradation(cells):
    """전류 요약과 초기 용량 변화율의 관계를 배치별 전체 표본에서 비교합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for axis, (batch, values) in zip(axes, cells.groupby("batch")):
        pairs = values[["I10_positive_mean", "QD_early_slope"]].dropna()
        axis.scatter(pairs.I10_positive_mean, pairs.QD_early_slope * 1e5, s=18)
        correlation = pairs.I10_positive_mean.corr(pairs.QD_early_slope)
        axis.axhline(0, color="gray", lw=0.6)
        axis.set(
            title=f"B{batch}: r={correlation:.3f}, n={len(pairs)}",
            xlabel="Cycle 10 positive sample mean I (A)",
            ylabel="Early QD slope (1e-5 Ah/cycle)",
        )
    eda.save_figure("12_current_degradation.png")


def main():
    cells = pd.read_csv(eda.RESULTS_DIR / "cell_features.csv")
    patterns, durations = load_charge_records()
    plot_current_patterns(patterns, cells)
    plot_charge_time_records(durations)
    plot_current_degradation(cells)
    save_charge_audit(durations, cells)


if __name__ == "__main__":
    main()
