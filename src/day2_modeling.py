"""Day2: Day1 사전 전략에 따른 셀 단위 학습, 선택, 외부 평가.

선택에는 B1 개발 셀만 사용합니다. Hold-out 확인 뒤 모델을 변경하지 않고,
전체 B1로 재학습한 하나의 모델로 B2/B3를 평가합니다.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.special import exp10
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results/cell_features.csv"
OUTPUT_DIR = PROJECT_DIR / "results/day2"
SEED = 42
TARGET_MAPE = 9.1
BASE_FEATURES = ("dq_logvar", "QD_early_slope", "Tavg_mean", "chargetime_median")
FEATURE_SETS = {
    "A": ("dq_logvar",),
    "B": BASE_FEATURES,
    "C": BASE_FEATURES + ("C1", "switch_pct", "C2"),
    "B_plus_QDdelta": BASE_FEATURES + ("QD100_minus_QD10",),
    "B_replace_slope": (
        "dq_logvar",
        "QD100_minus_QD10",
        "Tavg_mean",
        "chargetime_median",
    ),
    "B_charge_mean": ("dq_logvar", "QD_early_slope", "Tavg_mean", "chargetime_mean"),
    "B_no_charge": ("dq_logvar", "QD_early_slope", "Tavg_mean"),
}
ALLOWED_FEATURES = frozenset(
    feature for features in FEATURE_SETS.values() for feature in features
)


@dataclass(frozen=True)
class ModelSpec:
    feature_set: str
    estimator: str
    target: str = "raw"
    alpha: float = 0.0

    @property
    def name(self):
        return f"{self.feature_set}_{self.estimator}_{self.target}_a{self.alpha:g}"

    @property
    def features(self):
        # Dummy에도 입력 행 수 확인을 위한 컬럼 하나를 전달하되 사용하지 않습니다.
        return FEATURE_SETS[self.feature_set]

    @property
    def feature_count(self):
        return 0 if self.estimator == "dummy" else len(self.features)


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def candidate_specs():
    """Hold-out을 열기 전에 고정한 Day1 후보 39개."""
    candidates = [ModelSpec("A", "dummy")]
    candidates += [ModelSpec("A", "linear", target) for target in ("raw", "log10")]
    for feature_set in FEATURE_SETS:
        if feature_set == "A":
            continue
        for target in ("raw", "log10"):
            for alpha in (0.1, 1.0, 10.0):
                candidates.append(ModelSpec(feature_set, "ridge", target, alpha))
    return candidates


def build_model(spec):
    if not set(spec.features) <= ALLOWED_FEATURES:
        raise ValueError("사전 허용 목록 밖의 피처입니다.")
    estimators = {
        "dummy": lambda: DummyRegressor(strategy="median"),
        "linear": LinearRegression,
        "ridge": lambda: Ridge(alpha=spec.alpha),
    }
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("regressor", estimators[spec.estimator]()),
        ]
    )
    if spec.target == "log10":
        return TransformedTargetRegressor(
            regressor=pipeline, func=np.log10, inverse_func=exp10
        )
    return pipeline


def feature_matrix(cells, spec):
    return cells.loc[:, list(spec.features)].replace([np.inf, -np.inf], np.nan)


def load_cells(path=INPUT_PATH):
    cells = pd.read_csv(path)
    if cells.cell_id.duplicated().any():
        raise ValueError("셀 ID가 중복되었습니다.")
    missing = (ALLOWED_FEATURES | {"cycle_life", "batch", "policy"}) - set(
        cells.columns
    )
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    return cells


def labeled_batch(cells, batch):
    # 정답 결측은 제외하며 관측 사이클 수로 정답을 만들어내지 않습니다.
    return (
        cells.loc[
            (cells.batch == batch)
            & np.isfinite(cells.cycle_life)
            & (cells.cycle_life > 0)
        ]
        .copy()
        .reset_index(drop=True)
    )


def split_batch1(batch1):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_index, valid_index = next(splitter.split(batch1, groups=batch1.policy))
    development = batch1.iloc[train_index].reset_index(drop=True)
    holdout = batch1.iloc[valid_index].reset_index(drop=True)
    assert set(development.policy).isdisjoint(holdout.policy)
    assert set(development.cell_id).isdisjoint(holdout.cell_id)
    folds = list(GroupKFold(n_splits=5).split(development, groups=development.policy))
    for train, valid in folds:
        assert set(development.iloc[train].policy).isdisjoint(
            development.iloc[valid].policy
        )
    return development, holdout, folds


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if not np.isfinite(predicted).all():
        raise ValueError("유한하지 않은 예측값입니다.")
    return {
        "MAPE_pct": float(mean_absolute_percentage_error(actual, predicted) * 100),
        "MAE_cycles": float(mean_absolute_error(actual, predicted)),
        "RMSE_cycles": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def prediction_frame(cells, predicted, stage, candidate, fold=None):
    result = cells[
        ["cell_id", "batch", "policy", "cycle_life", "endpoint_quality_flag"]
    ].copy()
    result["prediction"] = predicted
    result["residual_cycles"] = result.prediction - result.cycle_life
    result["APE_pct"] = result.residual_cycles.abs() / result.cycle_life * 100
    result["stage"] = stage
    result["candidate"] = candidate
    result["fold"] = fold
    return result


def cross_validate(development, folds, specs):
    summaries, fold_records, predictions = [], [], []
    for spec in specs:
        scores = []
        for fold_number, (train_index, valid_index) in enumerate(folds, start=1):
            train, valid = development.iloc[train_index], development.iloc[valid_index]
            model = build_model(spec)
            model.fit(feature_matrix(train, spec), train.cycle_life)
            predicted = model.predict(feature_matrix(valid, spec))
            score = metrics(valid.cycle_life, predicted)
            scores.append(score["MAPE_pct"])
            fold_records.append(
                {
                    "candidate": spec.name,
                    "fold": fold_number,
                    "train_n": len(train),
                    "valid_n": len(valid),
                    **score,
                }
            )
            predictions.append(
                prediction_frame(valid, predicted, "CV", spec.name, fold_number)
            )
        std = float(np.std(scores, ddof=1))
        summaries.append(
            {
                "candidate": spec.name,
                **asdict(spec),
                "feature_count": spec.feature_count,
                "features": ", ".join(spec.features),
                "cv_mean_MAPE_pct": float(np.mean(scores)),
                "cv_std_MAPE_pct": std,
                "cv_SE_pct": std / np.sqrt(len(folds)),
            }
        )
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(fold_records),
        pd.concat(predictions, ignore_index=True),
    )


def select_one_se(cv_results, specs):
    best = cv_results.sort_values(["cv_mean_MAPE_pct", "candidate"]).iloc[0]
    threshold = float(best.cv_mean_MAPE_pct + best.cv_SE_pct)
    eligible = cv_results[cv_results.cv_mean_MAPE_pct <= threshold].copy()
    # Day1 고정 규칙: 적은 피처 → 단순 모델 → 원 타깃 → 강한 규제.
    eligible["model_order"] = eligible.estimator.map(
        {"dummy": 0, "linear": 1, "ridge": 2}
    )
    eligible["target_order"] = eligible.target.map({"raw": 0, "log10": 1})
    winner = eligible.sort_values(
        [
            "feature_count",
            "model_order",
            "target_order",
            "alpha",
            "cv_mean_MAPE_pct",
            "candidate",
        ],
        ascending=[True, True, True, False, True, True],
    ).iloc[0]
    selected = next(spec for spec in specs if spec.name == winner.candidate)
    return selected, {
        "best_mean_candidate": str(best.candidate),
        "best_mean_MAPE_pct": float(best.cv_mean_MAPE_pct),
        "best_SE_pct": float(best.cv_SE_pct),
        "one_SE_threshold_pct": threshold,
        "eligible_candidates": eligible.candidate.tolist(),
        "selected_candidate": selected.name,
        "selected_spec": asdict(selected),
        "features": list(selected.features),
        "selected_cv_mean_MAPE_pct": float(winner.cv_mean_MAPE_pct),
        "selected_cv_std_MAPE_pct": float(winner.cv_std_MAPE_pct),
        "rule": "fewest features, simplest model, raw target, stronger Ridge, lower CV MAPE",
    }


def residual_diagnostics(development, selected_oof):
    """같은 방향의 곡률이 각 fold에서 재현되는지 B1 OOF만으로 탐색합니다."""
    joined = selected_oof.merge(
        development[["cell_id", "dq_logvar"]], on="cell_id", validate="one_to_one"
    )
    centered_squared = (joined.dq_logvar - development.dq_logvar.median()) ** 2
    joined["squared_dq"] = centered_squared
    records = []
    for fold, group in joined.groupby("fold"):
        rho = float(spearmanr(group.squared_dq, group.residual_cycles).statistic)
        records.append(
            {"fold": int(fold), "n": len(group), "rho_squared_dq_residual": rho}
        )
    # 탐색적 진단이며, 새로운 모델을 자동 채택하기 위한 검정이 아닙니다.
    return records


def quality_sensitivity(development, folds, spec):
    """전 수명 품질 후보를 학습 fold에서만 제외해 같은 검증 셀로 비교합니다."""
    rows, prediction_rows = [], []
    for fold, (train_index, valid_index) in enumerate(folds, start=1):
        original_train = development.iloc[train_index]
        train = original_train.loc[~original_train.endpoint_quality_flag]
        valid = development.iloc[valid_index]
        model = build_model(spec)
        model.fit(feature_matrix(train, spec), train.cycle_life)
        predicted = model.predict(feature_matrix(valid, spec))
        rows.append(
            {
                "scenario": "train_only_endpoint_exclusion",
                "fold": fold,
                "train_original_n": len(original_train),
                "train_used_n": len(train),
                "valid_n": len(valid),
                **metrics(valid.cycle_life, predicted),
            }
        )
        prediction_rows.append(
            prediction_frame(valid, predicted, "quality_sensitivity", spec.name, fold)
        )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    group_records = []
    for flag, group in predictions.groupby("endpoint_quality_flag"):
        group_records.append(
            {
                "validation_endpoint_flag": bool(flag),
                "n": len(group),
                **metrics(group.cycle_life, group.prediction),
            }
        )
    return pd.DataFrame(rows), predictions, pd.DataFrame(group_records)


def make_performance_table(cv_mean, holdout_mape, batch2_mape, batch3_mape, cv_std):
    rows = [
        (
            "Train (Batch 1 CV)",
            cv_mean,
            f"5-fold 정책 그룹 CV; std={cv_std:.3f}%; 개발 셀만 사용",
        ),
        (
            "Valid (Batch 1 Hold-out)",
            holdout_mape,
            "모델 선택 후 1회 확인; 정책 그룹 분리",
        ),
        ("Test (Batch 2)", batch2_mape, "전체 B1 재학습 모델; 유효 라벨 39셀"),
        (
            "Gap (Train-Valid)",
            holdout_mape - cv_mean,
            "Valid - CV; 단위 %p; (+) 오차 증가",
        ),
        (
            "Gap (Valid-Test)",
            batch2_mape - holdout_mape,
            "B2 - Valid; 단위 %p; (+) 일반화 저하 가능성",
        ),
        (
            "Gap (Target-Test)",
            batch2_mape - TARGET_MAPE,
            "B2 - 9.1; 단위 %p; 동일 조건 재현 아님",
        ),
        (
            "Test (Batch 3)",
            batch3_mape,
            "선택 사항; B2와 동일한 최종 모델; 유효 라벨 44셀",
        ),
        (
            "Gap (Batch2-Batch3)",
            batch3_mape - batch2_mape,
            "B3 - B2; 단위 %p; (+) B3 오차 증가",
        ),
        (
            "Gap (Target-Test, Batch 3)",
            batch3_mape - TARGET_MAPE,
            "B3 - 9.1; 과제 비교 기준; 단위 %p",
        ),
    ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"])


def error_tables(predictions, cells, selected):
    external = predictions[predictions.stage.isin(["Test B2", "Test B3"])].copy()
    external["life_group"] = pd.cut(
        external.cycle_life,
        [-np.inf, 500, 1000, np.inf],
        labels=["<500", "500–1000", ">1000"],
        right=False,
    )
    # Day1 경계는 >1000; 정확히 1000인 경우 중간군으로 보정합니다.
    external.loc[external.cycle_life == 1000, "life_group"] = "500–1000"
    summaries = []
    for (stage, group_name), group in external.groupby(
        ["stage", "life_group"], observed=True
    ):
        summaries.append(
            {
                "stage": stage,
                "life_group": str(group_name),
                "n": len(group),
                "overprediction_n": int((group.residual_cycles > 0).sum()),
                **metrics(group.cycle_life, group.prediction),
            }
        )
    policy_rows = []
    for (stage, policy), group in external.groupby(["stage", "policy"]):
        policy_rows.append(
            {
                "stage": stage,
                "policy": policy,
                "n": len(group),
                "mean_residual_cycles": float(group.residual_cycles.mean()),
                **metrics(group.cycle_life, group.prediction),
            }
        )
    external.merge(
        cells[["cell_id"] + list(selected.features)],
        on="cell_id",
        validate="one_to_one",
    ).sort_values("APE_pct", ascending=False).to_csv(
        OUTPUT_DIR / "external_errors.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "errors_by_life_group.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(OUTPUT_DIR / "errors_by_policy.csv", index=False)
    ranges = []
    b1 = labeled_batch(cells, 1)
    for feature in selected.features:
        lo, hi = b1[feature].min(), b1[feature].max()
        for batch in (2, 3):
            values = labeled_batch(cells, batch)[feature]
            ranges.append(
                {
                    "batch": batch,
                    "feature": feature,
                    "B1_min": lo,
                    "B1_max": hi,
                    "external_min": values.min(),
                    "external_max": values.max(),
                    "outside_B1_n": int(((values < lo) | (values > hi)).sum()),
                    "missing_n": int(values.isna().sum()),
                }
            )
    pd.DataFrame(ranges).to_csv(OUTPUT_DIR / "feature_shift.csv", index=False)


def plot_results(cv_results, predictions):
    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)
    best_by_set = (
        cv_results.sort_values("cv_mean_MAPE_pct")
        .groupby("feature_set", sort=False)
        .head(1)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(best_by_set.candidate, best_by_set.cv_mean_MAPE_pct, color="0.45")
    ax.errorbar(
        best_by_set.cv_mean_MAPE_pct,
        best_by_set.candidate,
        xerr=best_by_set.cv_std_MAPE_pct,
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.set_xlabel("Group CV MAPE (%) — bars: mean; error bars: fold SD")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(figure_dir / "01_cv_comparison.png", dpi=150)
    plt.close(fig)
    stages = ["CV", "Hold-out B1", "Test B2", "Test B3"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, stage in zip(axes, stages):
        group = predictions[predictions.stage == stage]
        ax.scatter(group.cycle_life, group.prediction, color="0.3", s=25)
        lo = min(group.cycle_life.min(), group.prediction.min()) * 0.9
        hi = max(group.cycle_life.max(), group.prediction.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "--", color="0.6")
        ax.set(
            xlabel="Actual life (cycles)",
            ylabel="Predicted life (cycles)",
            title=f"{stage}, n={len(group)}",
            xlim=(lo, hi),
            ylim=(lo, hi),
        )
    fig.tight_layout()
    fig.savefig(figure_dir / "02_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, stage in zip(axes, ["Test B2", "Test B3"]):
        group = predictions[predictions.stage == stage]
        ax.scatter(group.cycle_life, group.residual_cycles, color="0.3")
        ax.axhline(0, color="0.6", linestyle="--")
        ax.set(
            xlabel="Actual life (cycles)",
            ylabel="Prediction - actual (cycles)",
            title=stage,
        )
    fig.tight_layout()
    fig.savefig(figure_dir / "03_external_residuals.png", dpi=150)
    plt.close(fig)


def run(output_dir=OUTPUT_DIR):
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    b1 = labeled_batch(cells, 1)
    development, holdout, folds = split_batch1(b1)
    specs = candidate_specs()
    manifest = {
        "seed": SEED,
        "holdout_policy_fraction": 0.2,
        "cv_folds": 5,
        "input_sha256": hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest(),
        "development_ids": development.cell_id.tolist(),
        "holdout_ids": holdout.cell_id.tolist(),
        "development_policies": sorted(development.policy.unique()),
        "holdout_policies": sorted(holdout.policy.unique()),
        "folds": [
            {
                "fold": fold,
                "train_ids": development.iloc[train].cell_id.tolist(),
                "valid_ids": development.iloc[valid].cell_id.tolist(),
            }
            for fold, (train, valid) in enumerate(folds, start=1)
        ],
        "candidates": [
            {"name": spec.name, **asdict(spec), "features": list(spec.features)}
            for spec in specs
        ],
        "nonlinear_models": "Deferred: exploratory residual checks do not establish reproducible nonlinearity.",
    }
    write_json(OUTPUT_DIR / "experiment_manifest.json", manifest)
    cv_results, fold_results, all_oof = cross_validate(development, folds, specs)
    selected, selection = select_one_se(cv_results, specs)
    # 선택 결과를 저장한 뒤 Hold-out 및 외부 평가로 진행합니다.
    write_json(OUTPUT_DIR / "model_selection.json", selection)
    cv_results.sort_values("cv_mean_MAPE_pct").to_csv(
        OUTPUT_DIR / "cv_candidates.csv", index=False
    )
    fold_results.to_csv(OUTPUT_DIR / "cv_folds.csv", index=False)
    all_oof.to_csv(OUTPUT_DIR / "cv_all_predictions.csv", index=False)
    selected_oof = all_oof[all_oof.candidate == selected.name].copy()
    write_json(
        OUTPUT_DIR / "residual_diagnostics.json",
        residual_diagnostics(development, selected_oof),
    )
    sensitivity_folds, sensitivity_pred, sensitivity_groups = quality_sensitivity(
        development, folds, selected
    )
    sensitivity_folds.to_csv(OUTPUT_DIR / "quality_sensitivity_folds.csv", index=False)
    sensitivity_pred.to_csv(
        OUTPUT_DIR / "quality_sensitivity_predictions.csv", index=False
    )
    sensitivity_groups.to_csv(
        OUTPUT_DIR / "quality_sensitivity_groups.csv", index=False
    )
    # 주 분석은 원본 B1을 모두 유지하며 민감도 결과로 모델 선택을 바꾸지 않습니다.
    development_model = build_model(selected)
    development_model.fit(feature_matrix(development, selected), development.cycle_life)
    holdout_prediction = development_model.predict(feature_matrix(holdout, selected))
    evaluation = {"Hold-out B1": metrics(holdout.cycle_life, holdout_prediction)}
    predictions = [
        selected_oof,
        prediction_frame(holdout, holdout_prediction, "Hold-out B1", selected.name),
    ]
    final_model = build_model(selected)
    final_model.fit(feature_matrix(b1, selected), b1.cycle_life)
    joblib.dump(
        {
            "model": final_model,
            "spec": asdict(selected),
            "features": list(selected.features),
            "training_ids": b1.cell_id.tolist(),
        },
        OUTPUT_DIR / "final_model.joblib",
    )
    for batch in (2, 3):
        test = labeled_batch(cells, batch)
        prediction = final_model.predict(feature_matrix(test, selected))
        stage = f"Test B{batch}"
        evaluation[stage] = metrics(test.cycle_life, prediction)
        predictions.append(prediction_frame(test, prediction, stage, selected.name))
    prediction_table = pd.concat(predictions, ignore_index=True)
    prediction_table.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    performance = make_performance_table(
        selection["selected_cv_mean_MAPE_pct"],
        evaluation["Hold-out B1"]["MAPE_pct"],
        evaluation["Test B2"]["MAPE_pct"],
        evaluation["Test B3"]["MAPE_pct"],
        selection["selected_cv_std_MAPE_pct"],
    )
    performance.to_csv(OUTPUT_DIR / "model_performance.csv", index=False)
    evaluation["CV pooled OOF"] = metrics(
        selected_oof.cycle_life, selected_oof.prediction
    )
    audit = {
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "matplotlib",
                "scikit-learn",
                "joblib",
            ]
        },
        "input_sha256": manifest["input_sha256"],
        "selected_candidate": selected.name,
        "raw_batch_counts": cells.groupby("batch").size().to_dict(),
        "labeled_batch_counts": {
            batch: len(labeled_batch(cells, batch)) for batch in (1, 2, 3)
        },
        "development_n": len(development),
        "holdout_n": len(holdout),
        "development_policy_n": int(development.policy.nunique()),
        "holdout_policy_n": int(holdout.policy.nunique()),
        "candidate_n": len(specs),
        "metrics": evaluation,
        "negative_predictions_n": int((prediction_table.prediction <= 0).sum()),
        "selection_uses": "Batch 1 development CV only; no hold-out/B2/B3 scores",
        "limitations": [
            "External labels were explored in required Day1 EDA.",
            "CV scores used for selection are not unbiased generalization estimates.",
            "Small non-independent group folds; fold SE is a heuristic.",
            "Raw labels retained; censored/continuation labels need author-level audit.",
        ],
    }
    write_json(OUTPUT_DIR / "evaluation_audit.json", audit)
    error_tables(prediction_table, cells, selected)
    plot_results(cv_results, prediction_table)
    print(
        json.dumps(
            {
                "selected": selected.name,
                "B1_development": len(development),
                "B1_holdout": len(holdout),
                "metrics": evaluation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return {
        "selection": selection,
        "audit": audit,
        "performance": performance,
        "cv_results": cv_results,
        "predictions": prediction_table,
    }


if __name__ == "__main__":
    run()
