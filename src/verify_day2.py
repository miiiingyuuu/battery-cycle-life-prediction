"""분할 누수·전처리 범위·지표·저장 모델의 재현성을 검증합니다."""

from collections import Counter
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

import day2_modeling as modeling


def verify():
    output = modeling.PROJECT_DIR / "results/day2"
    cells = modeling.load_cells()
    b1 = modeling.labeled_batch(cells, 1)
    development, holdout, folds = modeling.split_batch1(b1)
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    selection = json.loads((output / "model_selection.json").read_text())
    audit = json.loads((output / "evaluation_audit.json").read_text())
    cv = pd.read_csv(output / "cv_candidates.csv")
    fold_metrics = pd.read_csv(output / "cv_folds.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    performance = pd.read_csv(output / "model_performance.csv")
    checks = []

    assert cells.groupby("batch").size().to_dict() == {1: 46, 2: 47, 3: 46}
    assert [len(modeling.labeled_batch(cells, i)) for i in (1, 2, 3)] == [46, 39, 44]
    assert (
        manifest["input_sha256"]
        == hashlib.sha256(modeling.INPUT_PATH.read_bytes()).hexdigest()
    )
    checks.append("139셀 원본·46/39/44 유효 라벨·입력 해시 확인")

    assert manifest["development_ids"] == development.cell_id.tolist()
    assert manifest["holdout_ids"] == holdout.cell_id.tolist()
    assert set(development.policy).isdisjoint(holdout.policy)
    coverage = Counter()
    for train, valid in folds:
        assert set(development.iloc[train].policy).isdisjoint(
            development.iloc[valid].policy
        )
        assert set(development.iloc[train].cell_id).isdisjoint(
            development.iloc[valid].cell_id
        )
        assert set(development.iloc[valid].cell_id).isdisjoint(holdout.cell_id)
        coverage.update(development.iloc[valid].cell_id)
    assert coverage == Counter({cell_id: 1 for cell_id in development.cell_id})
    checks.append("Hold-out 및 5개 fold의 셀·정책 중복 없음; OOF 1회씩")

    selected, reproduced_selection = modeling.select_one_se(
        cv, modeling.candidate_specs()
    )
    assert selected.name == selection["selected_candidate"]
    assert np.isclose(
        reproduced_selection["one_SE_threshold_pct"], selection["one_SE_threshold_pct"]
    )
    assert len(cv) == 39
    checks.append("39개 후보 및 고정 1-SE 선택 규칙 재검증")

    # 외부 정답과 미래 설명 컬럼을 훼손해도 B1 개발 입력은 변하지 않습니다.
    poisoned = cells.copy()
    poisoned.loc[poisoned.batch != 1, "cycle_life"] = 1.0
    future_columns = ["last_QD", "knee", "n_summary", "n_cycles", "bad_QD_rows"]
    poisoned[future_columns] = 1e12
    poison_development, _, _ = modeling.split_batch1(
        modeling.labeled_batch(poisoned, 1)
    )
    pd.testing.assert_frame_equal(
        modeling.feature_matrix(development, selected),
        modeling.feature_matrix(poison_development, selected),
    )
    for spec in modeling.candidate_specs():
        assert set(spec.features) <= modeling.ALLOWED_FEATURES
        assert set(spec.features).isdisjoint(
            future_columns + ["cycle_life", "batch", "policy", "cell_id"]
        )
    checks.append(
        "외부 라벨·미래 정보 변경에 개발 입력 불변; 모든 후보 피처 허용 목록 확인"
    )

    # 입력 결측을 하나 만들어 imputer가 fold 학습 부분만 참조하는지 확인합니다.
    train_index, valid_index = folds[0]
    train, valid = (
        development.iloc[train_index].copy(),
        development.iloc[valid_index].copy(),
    )
    spec = modeling.ModelSpec("B", "ridge", alpha=1)
    train.loc[train.index[0], "Tavg_mean"] = np.nan
    valid["Tavg_mean"] = 1e9
    model = modeling.build_model(spec)
    model.fit(modeling.feature_matrix(train, spec), train.cycle_life)
    expected = modeling.feature_matrix(train, spec).median().to_numpy()
    np.testing.assert_allclose(model.named_steps["imputer"].statistics_, expected)
    before = model.named_steps["scaler"].mean_.copy()
    model.predict(modeling.feature_matrix(valid, spec))
    np.testing.assert_array_equal(before, model.named_steps["scaler"].mean_)
    checks.append(
        "결측 보완 통계는 fold train 중앙값; predict가 전처리 통계 변경하지 않음"
    )

    log_spec = modeling.ModelSpec("A", "linear", target="log10")
    log_model = modeling.build_model(log_spec)
    log_model.fit(modeling.feature_matrix(train, log_spec), train.cycle_life)
    log_prediction = log_model.predict(modeling.feature_matrix(valid, log_spec))
    np.testing.assert_allclose(
        log_prediction,
        10 ** log_model.regressor_.predict(modeling.feature_matrix(valid, log_spec)),
    )
    checks.append("로그 타깃 역변환 후 원 사이클 단위 평가 확인")

    bundle = joblib.load(output / "final_model.joblib")
    assert bundle["training_ids"] == b1.cell_id.tolist()
    for batch in (2, 3):
        test = modeling.labeled_batch(cells, batch)
        saved = (
            predictions[predictions.stage == f"Test B{batch}"]
            .set_index("cell_id")
            .loc[test.cell_id]
        )
        np.testing.assert_allclose(
            bundle["model"].predict(test[bundle["features"]]),
            saved.prediction,
            rtol=1e-12,
        )
        for key, value in modeling.metrics(saved.cycle_life, saved.prediction).items():
            assert np.isclose(value, audit["metrics"][f"Test B{batch}"][key])
    checks.append("B1 전체 46셀 재학습 및 저장 모델의 B2/B3 예측 재현")

    for stage, ids in [
        ("CV", development.cell_id),
        ("Hold-out B1", holdout.cell_id),
        ("Test B2", modeling.labeled_batch(cells, 2).cell_id),
        ("Test B3", modeling.labeled_batch(cells, 3).cell_id),
    ]:
        subset = predictions[predictions.stage == stage]
        assert set(subset.cell_id) == set(ids)
        assert not subset.cell_id.duplicated().any()
    for candidate, group in fold_metrics.groupby("candidate"):
        saved = cv.set_index("candidate").loc[candidate]
        assert np.isclose(saved.cv_mean_MAPE_pct, group.MAPE_pct.mean())
        assert np.isclose(saved.cv_std_MAPE_pct, group.MAPE_pct.std(ddof=1))
    assert np.isfinite(predictions.prediction).all()
    assert len(predictions) == 129
    table = performance.set_index("구분")["MAPE (%)"]
    assert np.isclose(
        table["Gap (Train-Valid)"],
        table["Valid (Batch 1 Hold-out)"] - table["Train (Batch 1 CV)"],
    )
    assert np.isclose(
        table["Gap (Valid-Test)"],
        table["Test (Batch 2)"] - table["Valid (Batch 1 Hold-out)"],
    )
    assert np.isclose(table["Gap (Target-Test)"], table["Test (Batch 2)"] - 9.1)
    assert np.isclose(
        table["Gap (Batch2-Batch3)"], table["Test (Batch 3)"] - table["Test (Batch 2)"]
    )
    checks.append("129개 예측의 모집단·fold 평균/표준편차·MAPE 및 Gap 산식 확인")

    sensitivity = pd.read_csv(output / "quality_sensitivity_predictions.csv")
    assert set(sensitivity.cell_id) == set(development.cell_id)
    assert not sensitivity.cell_id.duplicated().any()
    checks.append("끝 용량 후보 민감도도 원본과 같은 35개 검증 셀로 평가")
    result = {"status": "PASS", "check_count": len(checks), "checks": checks}
    modeling.write_json(output / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    verify()
