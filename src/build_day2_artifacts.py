"""실측 CSV/JSON을 바탕으로 README, 보고서, Day2 노트북을 생성합니다.

코드가 모두 동일한 노트북은 실행 출력을 보존합니다. 새 코드면 재실행이 필요합니다.
"""

from pathlib import Path
from copy import deepcopy
import json

import joblib
import nbformat
import numpy as np
import pandas as pd
from day2_eda_summary import eda_results_summary

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_DIR / "results/day2"
ASSIGNMENT = "https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb"


def markdown_table(frame, decimals=3):
    """추가 tabulate 의존성 없이 작은 결과 표를 출력합니다."""
    rows = [
        "| " + " | ".join(str(c) for c in frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        formatted = [
            f"{v:.{decimals}f}" if isinstance(v, (float, np.floating)) else str(v)
            for v in values
        ]
        rows.append("| " + " | ".join(v.replace("|", "/") for v in formatted) + " |")
    return "\n".join(rows)


def make_sections():
    audit = json.loads((RESULTS / "evaluation_audit.json").read_text())
    selection = json.loads((RESULTS / "model_selection.json").read_text())
    cv = pd.read_csv(RESULTS / "cv_candidates.csv")
    performance = pd.read_csv(RESULTS / "model_performance.csv")
    errors = pd.read_csv(RESULTS / "external_errors.csv")
    groups = pd.read_csv(RESULTS / "errors_by_life_group.csv")
    sensitivity = pd.read_csv(RESULTS / "quality_sensitivity_folds.csv")
    manifest = json.loads((RESULTS / "experiment_manifest.json").read_text())
    bundle = joblib.load(RESULTS / "final_model.joblib")
    scaler = bundle["model"].named_steps["scaler"]
    regressor = bundle["model"].named_steps["regressor"]
    coefficients = regressor.coef_ / scaler.scale_
    intercept = regressor.intercept_ - np.sum(
        regressor.coef_ * scaler.mean_ / scaler.scale_
    )
    best_by_set = (
        cv.sort_values("cv_mean_MAPE_pct").groupby("feature_set", sort=False).head(1)
    )
    dummy = cv[cv.estimator == "dummy"]
    comparison = pd.concat(
        [best_by_set[best_by_set.estimator != "dummy"], dummy]
    ).drop_duplicates("candidate")
    comparison = comparison[
        ["candidate", "feature_count", "cv_mean_MAPE_pct", "cv_std_MAPE_pct"]
    ]
    comparison.columns = [
        "각 세트의 최저 평균 후보 / Dummy",
        "피처 수",
        "CV MAPE (%)",
        "fold SD (%)",
    ]
    worst = pd.concat([errors[errors.batch == batch].head(3) for batch in (2, 3)])
    worst = worst[
        ["cell_id", "policy", "cycle_life", "prediction", "residual_cycles", "APE_pct"]
    ]
    holdout = audit["metrics"]["Hold-out B1"]
    b2, b3 = audit["metrics"]["Test B2"], audit["metrics"]["Test B3"]

    overview = """초기 100사이클까지의 신호로 총 Cycle Life를 예측해, ESS 셀의 열화 위험 선별과 점검 계획에 활용할 가능성을 조사한다. RUL을 직접 예측하는 모델이나 실제 ESS 배포 모델은 아니다.

- 데이터: MIT–Stanford Battery Dataset의 Kaggle version 1 배포본.
- 태스크: **Regression**, y=`cycle_life`(사이클). 분류는 수행하지 않는다.
- 논문·과제의 EOL 정의는 방전 용량이 초기 기준의 80%에 도달하는 시점이다. 이번 정답은 배포본의 `cycle_life`를 사용하며, 모든 라벨의 종료 조건·연속 셀 연결이 동일한지는 별도 감사가 필요하다. 관측 길이로 라벨을 재계산하지 않는다.
- 학습: Batch 1 (2017-05-12), 최종 시험: Batch 2 (2018-02-20).
- 추가 시험: Batch 3 (2018-04-12), 동일한 최종 모델 사용.
- 2018-04-03 extra varcharge 파일은 별도 충전 최적화 실험 데이터이므로 대상 배치에 포함하지 않는다. 이는 셀의 정책 문자열 VarCharge와 구분한다.
- 원시 셀 46/47/46개, 유효 라벨 46/39/44개. 라벨 결측 B2 8개·B3 2개는 평가에서 제외하며 관측 길이로 대체하지 않는다.
- 참가자: **울산 1반 박민규** — EDA, 피처 엔지니어링, 모델 개발, 평가, 문서 작성.
"""
    eda = """| Day1 관측 | Day2 구현 |
| --- | --- |
| B1/B2/B3 수명 중앙값 858.5/472/1,005.5회. B1에는 <500회 셀이 없고 B2에는 28개 | B1 내부 평가와 B2/B3 외부 평가를 구분하고 수명 구간별 오차 분석 |
| Qd 추이는 초기 증가와 이후 감소가 혼재. Knee는 탐색적 근사 | 10~100회 용량 기울기만 후보로 사용하고 전 수명 Knee는 X에서 제외 |
| ΔQ 로그 분산과 수명 Pearson r=-0.886/-0.902/-0.702 | 단일 피처 A 선형회귀를 핵심 기준 후보로 구현 |
| C-rate·온도·충전시간의 관계는 배치별 방향이 다름 | B/C의 추가 가치와 충전시간 평균/중앙값/제외를 동일 B1 CV로 비교 |
| 온도·용량 변화 피처의 강한 중복, 극단 충전시간 기록 | 대표 온도 Tavg, Ridge 규제, QD 변화량 추가/대체 실험 |

상세 그래프와 해석은 [Day1 보고서](DAY1-Design-울산_1반-박민규.md)와 [실행된 EDA 노트북](DAY1-EDA-울산_1반-박민규.ipynb)에 있다. 상관계수는 예측 성능이나 인과효과의 증거가 아니다.
"""
    eda += "\n" + eda_results_summary(markdown_table)
    features = """ΔQ(V)=Q100(V)-Q10(V), 실제 전압 2.0~3.5V의 1,000개 점에서 분산을 계산한다. 주 피처는 `dq_logvar=log10(max(Var(ΔQ), 1e-12))`다. 원본의 사이클 번호 10·100과 전압 축 정렬을 확인한 Day1 파생 파일을 사용한다.

| 세트 | 초기 피처 | 실험 목적 |
| --- | --- | --- |
| A | dq_logvar | 배치 간 방향이 일관된 핵심 신호 |
| B | A + QD_early_slope + Tavg_mean + chargetime_median | 용량·온도·충전시간의 추가 가치 |
| C | B + C1 + switch_pct + C2 | 충전 프로토콜의 추가 가치 |
| B_plus_QDdelta | B + QD100_minus_QD10 | 초기 용량 변화량 추가 |
| B_replace_slope | B의 slope를 QD100_minus_QD10으로 대체 | 중복 피처 대체 |
| B_charge_mean | B의 충전시간 중앙값을 평균으로 대체 | 극단 기록 영향 비교 |
| B_no_charge | B에서 충전시간 제외 | 해당 신호 의존성 확인 |

QD 기울기·온도·충전시간 집계 범위는 10~100회다. 용량은 고정 조건 0<QD<1.5Ah 안에서 기울기를 계산한다. C1·전환%·C2를 해석할 수 없는 VarCharge는 결측으로 두고 학습 fold의 중앙값으로 보완한다. 다른 값은 원본을 보존한다.

`cell_id`, `batch`, `policy`, `cycle_life`, `knee`, `last_QD`, `n_summary`, `n_cycles`, `bad_QD_rows`, `endpoint_quality_flag`는 X에서 제외한다. `policy`는 분할 그룹으로만 사용한다. 코드가 명시적인 초기 피처 허용 목록으로 X를 선택하므로 감사 컬럼이 섞이지 않는다.
"""
    pipeline = f"""1. B1의 23개 정책 그룹에 `GroupShuffleSplit(test_size=0.2, random_state=42)`를 적용한다. 개발 **35셀·18정책**, Hold-out **11셀·5정책**으로 나뉜다. 20%는 정책 수 기준이라 셀 비율은 달라진다.
2. 개발 35셀에서 `GroupKFold(5)`를 적용한다. 같은 셀·정책이 fold 양쪽이나 Hold-out과 겹치지 않는다. fold 검증 n은 8/8/7/6/6이다.
3. 각 fold train에만 median imputer → StandardScaler → 회귀 모델을 fit한다. 원 수명/로그 수명을 같은 fold로 비교하고, 로그 예측은 10의 거듭제곱으로 역변환해 원 사이클 단위에서 평가한다.
4. Dummy 중앙값 1개, A 선형회귀 2개, 여섯 다변수 세트의 Ridge α={{0.1,1,10}}×타깃 {{raw,log10}} 36개, **총 39개** 후보를 비교한다.
5. Day1 고정 1-SE 규칙으로 선택한다. 최저 평균 CV MAPE + 해당 후보 fold SD/√5 이내에서 피처 수 → 모델 단순성 → 원 타깃 → 강한 Ridge → CV 오차 순으로 선택한다.
6. 선택을 `model_selection.json`에 저장한 뒤 Hold-out을 확인한다. Hold-out 점수로 다시 선택하지 않는다. 전체 B1 **46셀**로 재학습하고 B2 **39셀**, B3 **44셀**을 평가한다.

분할·후보 목록은 `experiment_manifest.json`, 195개 후보-fold 결과는 `cv_folds.csv`, 모든 OOF 예측은 `cv_all_predictions.csv`에 저장한다. 입력 SHA-256은 `{manifest['input_sha256']}`다. 모델은 `final_model.joblib`에 저장한다.

과제상 B2/B3 EDA에서 외부 라벨을 이미 관찰했으므로 완전한 blind test라고 주장하지 않는다. Day2의 선택·전처리 학습에는 외부 평가 점수를 사용하지 않는다. CV는 후보 선택에도 사용했으므로 그 점수는 독립적인 최종 성능 추정치가 아니다. 1-SE는 그룹 fold가 독립이라는 보장이 없는 선택용 근사이며 신뢰구간이 아니다.
"""
    model_choice = f"""{markdown_table(comparison)}

**최종 모델: {selection['selected_candidate']}**, 초기 `dq_logvar` 하나를 사용하는 원 수명 선형회귀.

평균 최저 후보는 A 로그 타깃 선형회귀({selection['best_mean_MAPE_pct']:.3f}%)다. 그 후보 SE={selection['best_SE_pct']:.3f}%이므로 1-SE 허용 상한은 {selection['one_SE_threshold_pct']:.3f}%다. A 원 타깃의 {selection['selected_cv_mean_MAPE_pct']:.3f}%도 범위 안이며 Day1 규칙의 원 타깃 우선순위로 선택됐다. 더 많은 변수를 추가해도 단일 핵심 신호보다 평균 오차가 줄지 않았다.

전체 B1 학습 모델의 원 피처 단위 식은 `예측 수명 = {intercept:.3f} {coefficients[0]:+.3f} × dq_logvar`다. 로그 분산이 작을수록 예측 수명이 길다. 입력 범위를 벗어나는 외삽과 비물리적 예측에 주의해야 하며, 이번 저장 예측에는 비양수 값이 없었다. 성능을 좋게 보이게 하는 예측값 절단이나 사후 보정은 하지 않았다.

B1 OOF 잔차와 중심화한 dq_logvar 제곱의 Spearman ρ는 fold별 0.333/0.452/0.214/0.600/-0.257이었다. 작은 fold에서 크기·부호가 달라 일관된 곡률이 확인됐다고 보기 어렵다. 이는 정식 검정이 아니다. SVR·RF·ElasticNet은 이번 확정 실험에서 사용하지 않고, 별도 개발 데이터와 사전 탐색 범위를 갖춘 후속 실험으로 남긴다.

![피처 세트별 가장 낮은 평균 CV MAPE와 fold 표준편차](results/day2/figures/01_cv_comparison.png)
"""
    reporting = f"""MAPE=100×mean(|예측-실제|/실제)이며 낮을수록 좋다. **Train은 학습 데이터 적합 오차가 아니라 개발 셀의 CV 평균**이다. 아래 Gap 행은 오류율 차이인 **%p**이며, 양수는 뒤 평가의 오차 증가로 정의했다.

{markdown_table(performance)}

| 평가 | MAE / 회 | RMSE / 회 | R² |
| --- | --- | --- | --- |
| B1 Hold-out | {holdout['MAE_cycles']:.3f} | {holdout['RMSE_cycles']:.3f} | {holdout['R2']:.3f} |
| B2 | {b2['MAE_cycles']:.3f} | {b2['RMSE_cycles']:.3f} | {b2['R2']:.3f} |
| B3 | {b3['MAE_cycles']:.3f} | {b3['RMSE_cycles']:.3f} | {b3['R2']:.3f} |

- Train–Valid +5.943%p: 내부 선택 결과보다 Hold-out 오차가 크다. 작은 표본·그룹 구성·선택 편향과 과적합 가능성을 함께 고려하며 단독으로 원인을 확정하지 않는다.
- Valid–Test +18.385%p: B2 일반화 성능이 크게 저하됐다. 주요 원인 후보는 단수명 영역 부족과 피처–수명 관계의 배치 차이다.
- Target–Test +22.429%p: **논문 비교 목표 9.1%를 달성하지 못했다.** 데이터 배치 날짜, 연속 셀 연결·제외 규칙, 분할, 피처와 모델이 달라 동일 조건 논문 재현이라고 해석할 수 없다.
- B3–B2 -18.939%p: B3의 MAPE는 낮지만 B3 장수명 셀에는 큰 과소 예측이 있다. 평균 지표만으로 실제 운용 적합성을 판단하지 않는다. B3 목표 차이는 과제의 9.1% 비교 기준이며 원논문의 B3 고유 성능을 재현한 값이 아니다.

OOF 전체 셀을 한 번에 계산한 MAPE는 7.290%로, 과제 표의 fold 평균 7.201%와 다르다. fold 크기가 달라 가중치가 다르며 표에는 요구대로 fold 평균을 사용했다.

![실제 수명과 예측 수명](results/day2/figures/02_actual_vs_predicted.png)
"""
    error_analysis = f"""{markdown_table(worst)}

B2에서 APE가 가장 큰 b2c6는 실제 393회인데 약 675회로 예측해 **71.647%** 오차가 났다. b2c15도 같은 `3.6C(9%)-5C` 정책이며, b2c18은 다른 정책에서 큰 오차를 보였다. 일부 정책과 단수명 구간에서 수명이 과대 예측되지만 정책 자체가 원인이라고 단정할 수 없다.

B3의 b3c38은 실제 1,935회인데 약 1,026회로 예측해 **909회 과소 예측**됐다. B1 최대 수명이 1,227회여서 이런 장수명 영역을 학습하지 못했다. `newstructure`의 Qdlin 측정 절차 차이도 원인 후보이며 현재 분산 피처만으로 해결됐다고 주장하지 않는다.

{markdown_table(groups[['stage', 'life_group', 'n', 'overprediction_n', 'MAPE_pct', 'MAE_cycles']])}

B2 단수명(<500) 28셀 **모두 과대 예측**, MAPE 36.749%다. B3 장수명(>1000) 23셀 **모두 과소 예측**이다. B1 dq_logvar 범위는 -5.015~-3.350이고 B2 11/39셀·B3 1/44셀이 이를 벗어난다. 그러나 최대 B2 오차 셀 b2c6는 범위 안에 있으므로 입력 범위 이탈만으로 실패를 설명할 수 없다. B1에서 배운 조건부 관계가 배치 간 그대로 유지되지 않았을 가능성을 추가 검증해야 한다.

`external_errors.csv`, `errors_by_policy.csv`, `errors_by_life_group.csv`, `feature_shift.csv`에 전체 결과가 있다. 후속 개선은 신규 독립 개발 데이터의 단수명/장수명 보강, 라벨 연결·검열 감사, 측정 절차 정합, 시간 가중 전류·곡선 피처 검증 순으로 진행한다. 이미 확인한 B2 점수에 맞춰 모델이나 삭제 규칙을 바꾼 결과를 새로운 test 성능으로 제시하지 않는다.

![외부 배치 잔차](results/day2/figures/03_external_residuals.png)
"""
    quality = f"""주 분석은 원본 B1 46셀을 유지한다. `endpoint_quality_flag`는 끝 QD>0.885Ah인 확인 후보이며 미완료/잘못된 라벨이 확정된 것은 아니다. 입력 피처나 배포 시 판별 규칙으로 사용하지 않는다.

고정 fold의 **학습 셀에서만** 후보를 제외한 보조 모델을 같은 35개 OOF 검증 셀로 평가했다. 원본 학습 CV 평균은 {selection['selected_cv_mean_MAPE_pct']:.3f}%, 후보 제외 학습은 {sensitivity.MAPE_pct.mean():.3f}%다. 원본과 검증 분모를 유지했으며, 이 결과로 최종 선택·B2/B3 모델을 변경하지 않았다.

검증 셀 중 품질 후보 포함/미포함 n은 7/28이다. 제외 학습 모델의 해당 MAPE는 4.297/8.360%이며 작은 집단의 기술 통계다. 최종 46셀 전체 학습과 보조 35셀 CV 실험의 학습 모집단을 구분한다. 종료 사유와 원저자 연속 셀 연결을 확인하기 전에는 일괄 삭제나 관측 길이 기반 라벨 수정이 적절하지 않다.
"""
    domain = """ESS 운영에서 이런 모델은 초기 사용 이력으로 상대적 열화 위험을 선별하고, 점검 우선순위·예비 셀 확보·교체 예산 검토를 돕는 후보 도구다. 현재 모델은 Batch 2의 단수명 셀을 일관되게 과대 예측하므로 교체 시점을 자동 결정하는 근거로 사용할 수 없다. 과대 예측은 늦은 점검·교체로, 과소 예측은 불필요하게 이른 교체와 비용 증가로 이어질 수 있다.

실험실 단일 셀의 총 사이클 수를 예측했으며 현재 SOH, 팩 수명, 잔여 달력 시간이나 정확한 고장 날짜를 계산한 것이 아니다. 팩의 셀 편차·열관리·부하 프로파일·SOC 운용 구간·달력 열화·BMS 센서 품질을 추가로 반영해야 한다. 100사이클까지 관찰해야 한다는 적용 시점 제약도 있다.

배포 전에는 현장 BMS 데이터로 시간·사이트·팩 단위 외부 검증, 신규 배치에서의 보정과 불확실성 평가, 비용을 반영한 과대/과소 예측 평가가 필요하다. 모델을 업데이트하면 새 평가셋을 확보하고, 초기 피처 분포·결측률·현장 오차를 모니터링한다. 예측을 실제 SOH 측정·안전 기준·운영자의 검토와 함께 사용해야 한다.

### 모니터링·재학습 운영 계획

아래 주기와 수치는 **현장 검증 전의 제안 기준**이다. 이번 과제에서 효과를 입증하거나 자동화한 운영 규칙이 아니다. 현재 모델은 B2 목표 미달과 단수명 과대 예측 때문에 연구용 참고 단계이며, 실제 ESS 교체 결정을 자동 수행하지 않는다.

| 주기 | 확인 내용 | 기록 및 조치 |
| --- | --- | --- |
| 매주 | 초기 피처 결측률, 기준 개발 데이터 범위를 벗어난 셀 비율, 센서·프로토콜 변경 | 모델/데이터 버전별 기록. 이상 신호는 데이터·측정 절차를 먼저 감사 |
| 매월 | 새로 EOL이 확인된 셀의 MAPE·MAE, 단수명 과대 예측 및 장수명 과소 예측 | 신규 유효 라벨 30셀 이상일 때 평가 창을 구성. 부족하면 누적하고 성능 저하를 확정하지 않음 |
| 분기마다 | 새로운 라벨·대표성·검열 상태, 최근 오류와 재학습 필요성 | 후보 재학습 여부를 검토하며 기존 모델을 자동 교체하지 않음 |

재학습 검토 촉발 조건은 ① 주간 결측률>5% 또는 기준 범위 이탈 비율>10%가 2회 연속 발생, ② 새로운 현장 기준 성능보다 월간 MAPE가 5%p 이상 증가하는 평가 창이 2회 연속 발생, ③ 센서·곡선 측정 절차·셀 종류·충전 프로토콜이 변경되는 경우로 제안한다. 현장 기준 성능은 최초 독립 현장 검증에서 설정하며, B1 CV 7.201%나 논문 9.1%를 현장 정상값으로 가정하지 않는다. 기준 범위는 초기 승인 개발 데이터에서 고정하고 운영 데이터로 자동 갱신하지 않는다.

경보가 나면 박민규가 데이터·라벨 원인을 검토하고, 실제 배포 단계에서는 현장 운영 책임자가 사용 제한과 갱신을 승인하도록 한다. EOL 라벨은 늦게 확보되므로 입력 드리프트 경보와 성능 경보를 구분한다. 라벨 30셀은 실무 검토를 시작하기 위한 제안 최소량이며 통계적 신뢰성을 보장하지 않는다.

후보를 재학습할 때는 신규 개발 데이터와 시간·사이트·팩·정책 그룹을 분리한 **새 Hold-out**을 먼저 고정한다. 과제 B2/B3를 새 시험처럼 재사용하지 않는다. 새 모델 채택은 같은 새 검증 셀에서 기준 모델보다 MAPE가 낮고 단수명 과대 예측 위험이 악화되지 않으며, 충분한 표본·오차 불확실성·현장 BMS 검증을 운영 책임자가 확인한 경우에만 검토한다. 개선을 확인하지 못하면 기존 연구 버전을 유지하고 적용 범위를 제한한다. 모델·데이터·평가 날짜와 승인 사유를 기록해 이전 버전으로 되돌릴 수 있게 한다.
"""
    reproduction = """## 환경 설정과 실행

Python 3.14.7에서 아래 고정 버전으로 실제 실행했다. 같은 버전의 Python을 사용하면 재현 환경 차이를 줄일 수 있다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
python src/day2_modeling.py
python src/verify_day2.py
```

위 명령은 **README가 있는 mini-project 폴더 또는 독립 저장소 루트**에서 실행한다. 저장된 `results/cell_features.csv`만으로 Day2를 재실행할 수 있다. 원본부터의 재현은 [data/README.md](data/README.md)에 안내했다. 기존 Day1 PDF 생성은 macOS AppleGothic 폰트를 사용한다.

[Day2 노트북](DAY2-Modeling-울산_1반-박민규.ipynb)은 같은 환경을 커널로 선택해 위에서 아래로 실행한다. 저장된 출력으로 결과를 바로 확인할 수 있다. 재실행은 고정된 동일 실험의 재현이며 외부 점수를 보고 선택 규칙을 바꾸는 절차가 아니다. 생성기는 코드 원문이 모두 같으면 기존 실행 출력을 보존한다. 코드가 바뀌거나 새 노트북이면 출력이 초기화되므로 이후 전체 셀을 실행해야 한다.

Day1 EDA의 버전 감사는 PDF 생성용 `reportlab`·`pymupdf`가 없어도 실행된다. 미설치 패키지는 `results/audit.json`에 `null`과 누락 목록으로 기록한다. PDF를 다시 생성할 때는 전체 의존성이 필요하다. 설치한 Python과 노트북 커널이 같은 환경인지 확인하려면 노트북에서 `import sys; print(sys.executable)`로 경로를 확인한다.

## 파일 구조

```text
├── README.md
├── DAY1-Design-울산_1반-박민규.md
├── DAY1-EDA-울산_1반-박민규.ipynb
├── DAY2-Modeling-울산_1반-박민규.ipynb
├── DAY2-Report-울산_1반-박민규.md
├── DAY2-Requirements-Check.md
├── DAY2-Rubric-Review.md
├── data/README.md
├── src/
│   ├── day1_eda.py
│   ├── day1_charge_analysis.py
│   ├── day2_eda_summary.py
│   ├── day2_modeling.py
│   ├── verify_day2.py
│   └── build_day2_artifacts.py
├── results/
│   ├── cell_features.csv
│   └── day2/
│       ├── experiment_manifest.json
│       ├── model_selection.json
│       ├── cv_candidates.csv
│       ├── cv_folds.csv
│       ├── model_performance.csv
│       ├── predictions.csv
│       ├── final_model.joblib
│       ├── evaluation_audit.json
│       ├── verification.json
│       └── figures/
├── requirements.txt
└── requirements-notebook.txt
```

`src/verify_day2.py`는 정책·셀 중복, OOF 분모, fold train 전처리, 미래 정보 제외, 로그 역변환, 저장 모델 재현, 지표·Gap을 점검한다. 공식 채점이나 통계적 일반화 보증을 대신하지 않는다.

실행 노트북의 코드 셀 9개와 PNG 출력 3개는 오류 없이 저장했다. 별도 폴더에 제출 ZIP을 풀어 원본 MAT 없이 학습·검증을 다시 실행한 성능 차이는 0이었다. 기록은 `results/day2/notebook_execution.json`, `clean_reproduction.json`에서 확인한다.
"""
    references = f"""- [DS Mini Project: Task·Deliverables·평가 항목]({ASSIGNMENT})
- [DS Course Wrap-up](https://actually-war-1ea.notion.site/DS-Course-Wrap-up-27f7f4c866938053a154c817021f774f)
- [Severson et al. (2019), Nature Energy 4, 383–391](https://www.nature.com/articles/s41560-019-0356-8)
- [논문 저자 데이터 처리·모델 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- [Kaggle 배포본](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)
"""
    return {
        "overview": overview,
        "eda": eda,
        "features": features,
        "pipeline": pipeline,
        "model_choice": model_choice,
        "reporting": reporting,
        "errors": error_analysis,
        "quality": quality,
        "domain": domain,
        "reproduction": reproduction,
        "references": references,
    }


def write_documents(sections):
    headings = [
        ("프로젝트 개요", "overview"),
        ("Day1 EDA → 구현", "eda"),
        ("피처 설계", "features"),
        ("데이터 분리와 Pipeline", "pipeline"),
        ("모델 비교와 최종 선택", "model_choice"),
        ("성능 결과와 Gap", "reporting"),
        ("오류 분석", "errors"),
        ("품질 후보 민감도", "quality"),
        ("ESS 도메인 해석과 한계", "domain"),
    ]
    report = "# Day2 모델 개발 및 평가\n\n울산 1반 박민규\n\n"
    report += "\n\n".join(
        f"## {heading}\n\n{sections[key]}" for heading, key in headings
    )
    report += (
        "\n\n"
        + sections["reproduction"]
        + "\n\n## 참고문헌\n\n"
        + sections["references"]
    )
    (PROJECT_DIR / "DAY2-Report-울산_1반-박민규.md").write_text(
        report, encoding="utf-8"
    )
    # 제출용 README는 직접 관리하며 보고서 재생성 시 덮어쓰지 않습니다.


def write_review():
    requirements = """# Day2 요구사항 최종 대조

기준: [DS Mini Project](https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb)의 Task → Day2, Deliverables, 산출물 평가 항목. 울산 1반 박민규.

| 요구사항 | 상태 | 실제 증거 |
| --- | --- | --- |
| Day1 전략 기반 모델 개발 | 충족 | A/B/C, 용량 변화량 추가·대체, 충전시간 평균·중앙값·제외; 39개 후보 |
| Regression 또는 Classification 택1 | 충족 | y=cycle_life 회귀; 분류 지표는 해당 없음 |
| Batch 1 학습·Batch 2 시험 | 충족 | B1 46개 전체 재학습; B2 유효 라벨 39개; evaluation_audit.json |
| Batch 1 CV 평균 | 충족 | 개발 35셀 정책 GroupKFold 5개; cv_folds.csv |
| Batch 1 Hold-out | 충족 | 별도 11셀·5정책; experiment_manifest.json |
| 셀·충전 프로토콜 분리 | 충족 | Hold-out/CV 정책 중복 0; verify_day2.py |
| 누수 없는 전처리·핵심 변수 | 충족 | 초기 피처 허용 목록, fold train median/scaler, 로그 역변환 |
| MAPE 및 필수 6행 보고 형식 | 충족 | results/day2/model_performance.csv와 README |
| Train–Valid / Valid–Test / Target–Test | 충족 | 각 Gap 산식·%p·해석 명시 |
| 논문 9.1%와 Batch 2 비교 | 충족 | 실제 31.529%, +22.429%p; 목표 미달·조건 차이 명시 |
| Batch 3 추가 검증 | 선택 항목 충족 | 동일 모델 44셀; MAPE 12.591%, B3−B2 -18.939%p |
| Batch 3 측정·분포·이상치 주의 | 충족 | newstructure·장수명 외삽·라벨 감사; 일괄 삭제 안 함 |
| EDA 요약 및 피처·모델 선택 근거 | 충족 | README 및 Day2 보고서의 Day1→구현·39개 비교·1-SE 선택 |
| README 샘플의 독립 EDA 요약 | 보완 완료 | 수명 비율·분포 형태·Knee·장단 ΔQ·정책 평균을 실측 CSV에서 직접 요약 |
| Warm-up 재학습 주기·조건 | 계획 구체화 완료 | 주간/월간/분기 점검, 촉발 조건, 신규 Hold-out·승인·버전 관리. 현장 미검증 제안이며 운영 자동화는 미구현 |
| 오류 분석 | 충족 | 최악 셀, 수명 구간·정책·입력 범위별 결과 및 개선 방향 |
| ESS 활용 의미·한계 | 충족 | 점검·교체 위험, 단수명 과대 예측, 현장 검증·팩/달력 열화 |
| 환경·파일 구조·재현 명령·참고문헌·팀 | 충족 | README, requirements, data/README.md |
| 실행 가능한 코드·노트북 | 충족 | day2_modeling.py, verify_day2.py, DAY2-Modeling 노트북; notebook_execution.json |
| GitHub public 링크 | 충족 | [과제 공개 저장소](https://github.com/miiiingyuuu/battery-cycle-life-prediction); main에 제출 파일 업로드 완료 |

**검토 범위는 저장소의 분석·코드·보고 내용과 공개 링크다.** 해당 내용은 충족하며, 제출 시각과 외부 제출 절차는 검토 범위에서 제외한다. 목표 9.1%의 달성은 실패했지만, 성능 비교·Gap 보고 요구는 충족했다. 낮은 test 성능을 숨기거나 test에 맞춰 선택을 변경하지 않는다.

`results/day2/verification.json`은 분할·피처·전처리·지표·모델 재현을 검증한 결과다. 노트북 실제 실행 검사는 `results/day2/notebook_execution.json`에 기록한다.

최종 검증 결과: 노트북 9개 코드 셀 실행·PNG 출력 3개·오류 0개. ZIP을 별도 폴더에 풀어 원본 MAT 없이 학습/검증을 실행했으며 성능 차이는 0이었다(`clean_reproduction.json`).
"""
    (PROJECT_DIR / "DAY2-Requirements-Check.md").write_text(
        requirements, encoding="utf-8"
    )
    review = """# Day2 채점표 기반 자체 평가

울산 1반 박민규. 공식 점수가 아니며, 과제 페이지의 20/40/20/20 배점에 따라 로컬 산출물을 보수적으로 검토했다. 과제용 공개 저장소에 제출 파일이 업로드되어 있다. 외부 제출 절차와 제출 시각은 자체 평가 범위에서 제외한다.

| 평가 항목 | 배점 | 자체 점수 | 충족 근거 | 감점 가능성 |
| --- | --- | --- | --- | --- |
| 전략 → 구현 반영 | 20 | 20 | Day1 A/B/C·대체 세트, raw/log 타깃, 고정 1-SE 선택, fold 내 품질 민감도 모두 구현 | 추가 피처가 성능을 개선하지 않았다는 결과도 그대로 해석 |
| Pipeline 개발 | 40 | 38 | 정책 그룹 CV/Hold-out, fold train 전처리, 명시적 초기 X, 전체 B1 재학습, B2/B3 시험, 저장 모델 재현 | 개발 35셀 대비 39후보 선택 불확실성, 연속 셀·검열 라벨 정합 미확정 |
| 성능 리포팅 및 해석 | 20 | 18 | 6개 필수 행·3개 선택 행, MAPE/%p 산식, 9.1% 비교, 보조 지표·최악 셀·그룹 오차 | B2 31.529%로 목표 미달, 논문과 조건 일치 및 저하 원인의 독립 검증 부족 |
| 분석 결과 해석 | 20 | 18 | 단수명 과대 예측의 늦은 교체 위험, 장수명 과소 예측 비용, 현장·팩·달력 열화 한계와 개선 계획 | 실제 ESS 데이터와 비용·불확실성 검증은 수행하지 않음 |
| 합계 | 100 | **94** | 로컬 분석·문서 기준 | 실제 채점은 강사 판단에 따름 |

## 판정

- Day2 분석·코드·보고 형식과 README 내용 및 공개 저장소 업로드는 충족했다. 논문 목표 성능의 달성을 뜻하지 않는다.
- 목표 MAPE 9.1%는 미달이다. 채점표에는 목표 달성 자체의 별도 배점이 명시되지 않았으므로 위 점수는 목표 미달에 대한 공식 감점 공식을 뜻하지 않는다. 일반화와 근거의 부족을 보수적으로 반영한 자체 판단이다.
- 94점은 저장소 내용에 대한 비공식 자체 추정치다. 외부 제출 절차와 제출 시각은 평가하지 않는다.
- 한계의 정직한 기재는 완료했지만, 한계를 해소한 것은 아니다. 추가 개발 데이터 없이 이미 본 Batch 2에 맞춘 재튜닝으로 목표를 맞추는 것은 적절한 최종 평가가 아니다.

## 실행 검증

분할·전처리·피처·로그 역변환·129개 예측·MAPE/Gap·저장 모델 재현 검사는 `verification.json`에, 노트북 실행 검사는 `notebook_execution.json`에 있다. 이 검증은 통계적 일반화나 실제 ESS 배포 적합성의 보증이 아니다.
"""
    (PROJECT_DIR / "DAY2-Rubric-Review.md").write_text(review, encoding="utf-8")


def write_notebook(sections):
    cells = []
    md = lambda text: cells.append(nbformat.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbformat.v4.new_code_cell(text))
    md(
        "# Day2 모델 개발 및 평가\n\n울산 1반 박민규\n\n초기 100회로 전체 수명을 예측하는 회귀 실습입니다. 이 노트북은 고정 실험을 실제 재실행하고 결과를 표시합니다. 외부 시험 점수를 보고 후보·분할·처리를 변경하지 않습니다."
    )
    md("## 1. 환경과 데이터\n\n" + sections["overview"])
    code("""from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image

PROJECT = Path.cwd()
if not (PROJECT / 'src/day2_modeling.py').exists():
    PROJECT = PROJECT / 'mini-project'
if not (PROJECT / 'src/day2_modeling.py').exists():
    raise FileNotFoundError('저장소 루트 또는 mini-project 폴더에서 실행하세요.')
sys.path.insert(0, str(PROJECT / 'src'))
import day2_modeling as modeling
import verify_day2
RESULTS = PROJECT / 'results/day2'
cells = modeling.load_cells()
display(cells.groupby('batch').agg(raw_n=('cell_id', 'size'), labeled_n=('cycle_life', 'count')))
print('Python:', sys.version.split()[0])""")
    md(
        "## 2. Day1에서 사용할 정보를 고르기\n\n"
        + sections["eda"]
        + "\n"
        + sections["features"]
    )
    code("""display(pd.DataFrame([
    {'set': name, 'features': ', '.join(features)}
    for name, features in modeling.FEATURE_SETS.items()
]))""")
    md("## 3. 셀·정책 그룹으로 나누기\n\n" + sections["pipeline"])
    code("""b1 = modeling.labeled_batch(cells, 1)
development, holdout, folds = modeling.split_batch1(b1)
display(pd.DataFrame([
    {'split': 'B1 development', 'cells': len(development), 'policies': development.policy.nunique()},
    {'split': 'B1 hold-out', 'cells': len(holdout), 'policies': holdout.policy.nunique()},
]))
display(pd.DataFrame([
    {'fold': i, 'train_n': len(train), 'valid_n': len(valid),
     'policy_overlap': len(set(development.iloc[train].policy) & set(development.iloc[valid].policy))}
    for i, (train, valid) in enumerate(folds, 1)
]))""")
    md(
        "## 4. 고정한 전체 실험 실행\n\n아래 셀에서 CV 선택 → Hold-out → 전체 B1 재학습 → B2/B3 평가를 실행합니다. 전처리와 타깃 변환을 포함한 코드의 원문은 `src/day2_modeling.py`에서 확인할 수 있습니다."
    )
    code("""run_result = modeling.run()
verification = verify_day2.verify()""")
    md("## 5. 모델 비교와 선택 근거\n\n" + sections["model_choice"])
    code("""cv = pd.read_csv(RESULTS / 'cv_candidates.csv')
display(cv.sort_values('cv_mean_MAPE_pct').head(12))
selection = json.loads((RESULTS / 'model_selection.json').read_text())
display(pd.Series(selection))
display(Image(filename=str(RESULTS / 'figures/01_cv_comparison.png')))""")
    md("## 6. 성능표와 Gap 이해하기\n\n" + sections["reporting"])
    code("""performance = pd.read_csv(RESULTS / 'model_performance.csv')
display(performance)
audit = json.loads((RESULTS / 'evaluation_audit.json').read_text())
display(pd.DataFrame(audit['metrics']).T)
display(Image(filename=str(RESULTS / 'figures/02_actual_vs_predicted.png')))""")
    md("## 7. 틀린 예측을 직접 확인하기\n\n" + sections["errors"])
    code("""errors = pd.read_csv(RESULTS / 'external_errors.csv')
for batch in (2, 3):
    print(f'Batch {batch}: APE가 큰 셀')
    display(errors[errors.batch == batch].head(5))
display(pd.read_csv(RESULTS / 'errors_by_life_group.csv'))
display(pd.read_csv(RESULTS / 'feature_shift.csv'))
display(Image(filename=str(RESULTS / 'figures/03_external_residuals.png')))""")
    md("## 8. 품질 후보 삭제가 답인지 확인하기\n\n" + sections["quality"])
    code(
        """display(pd.read_csv(RESULTS / 'quality_sensitivity_folds.csv'))
display(pd.read_csv(RESULTS / 'quality_sensitivity_groups.csv'))
display(pd.DataFrame(json.loads((RESULTS / 'residual_diagnostics.json').read_text())))"""
    )
    md(
        "## 9. ESS 해석과 검증\n\n"
        + sections["domain"]
        + "\n\n과제 공개 저장소: [https://github.com/miiiingyuuu/battery-cycle-life-prediction](https://github.com/miiiingyuuu/battery-cycle-life-prediction). 분석·코드·보고 내용은 요구사항 대조 문서에서 확인할 수 있습니다."
    )
    code(
        """display(pd.DataFrame({'passed_check': verification['checks']}))
assert verification['status'] == 'PASS'
assert len(run_result['predictions']) == 129
print('모델·성능 재현 검증 완료. Batch 2 MAPE는 31.529%로 논문 목표 9.1%에 미달합니다.')"""
    )
    md("## 참고문헌\n\n" + sections["references"])
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )
    path = PROJECT_DIR / "DAY2-Modeling-울산_1반-박민규.ipynb"
    if path.exists():
        previous = nbformat.read(path, as_version=4)
        old_code = [cell for cell in previous.cells if cell.cell_type == "code"]
        new_code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if [cell.source for cell in old_code] == [cell.source for cell in new_code]:
            for old_cell, new_cell in zip(old_code, new_code):
                new_cell.outputs = deepcopy(old_cell.outputs)
                new_cell.execution_count = old_cell.execution_count
                new_cell.metadata = deepcopy(old_cell.metadata)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main():
    sections = make_sections()
    write_documents(sections)
    write_review()
    write_notebook(sections)
    print(
        "Day2 보고서·검토 문서·노트북 생성 완료. README는 유지합니다. 코드 변경 시 노트북을 전체 실행하세요."
    )


if __name__ == "__main__":
    main()
