"""Day1 실측 결과를 README·보고서·노트북의 독립 요약으로 구성합니다."""

from pathlib import Path
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def eda_results_summary(table):
    stats = pd.read_csv(RESULTS / "batch_stats.csv")
    diagnostics = pd.read_csv(RESULTS / "diagnostic_summary.csv")
    cells = pd.read_csv(RESULTS / "cell_features.csv")
    policies = pd.read_csv(RESULTS / "policy_stats.csv")
    distributions, knees, policy_examples = [], [], []
    for row in stats.itertuples():
        distributions.append(
            {
                "배치": f"B{row.batch}",
                "유효 라벨 n": row.label_n,
                "수명 범위 / 회": f"{row.min:.0f}~{row.max:.0f}",
                "중앙값 / 회": row.median,
                "단수명 <500": f"{row.short_lt500}/{row.label_n} ({100*row.short_lt500/row.label_n:.2f}%)",
                "장수명 >1000": f"{row.long_gt1000}/{row.label_n} ({100*row.long_gt1000/row.label_n:.2f}%)",
            }
        )
        valid = policies[
            (policies.batch == row.batch) & (policies["count"] > 0)
        ].sort_values("mean")
        for label, policy in [("최저", valid.iloc[0]), ("최고", valid.iloc[-1])]:
            policy_examples.append(
                {
                    "배치": f"B{row.batch}",
                    "평균 극단": label,
                    "정책": policy.policy,
                    "n": int(policy["count"]),
                    "평균 수명 / 회": float(policy["mean"]),
                }
            )
    for row in diagnostics.itertuples():
        knees.append(
            {
                "배치": f"B{row.batch}",
                "적합 n": row.knee_fit_n,
                "탐색 기준 충족 n": row.knee_supported,
                "Knee 후보 중앙값 / 회": row.knee_median,
            }
        )
    short = cells[(cells.batch == 2) & (cells.cycle_life < 500)]
    long = cells[(cells.batch == 2) & (cells.cycle_life > 1000)]
    relative = pd.read_csv(RESULTS / "relative_delta_groups.csv")
    relative_rows = []
    for batch, groups in relative.groupby("batch"):
        lower = groups[groups.group == "Bottom quartile"].iloc[0]
        upper = groups[groups.group == "Top quartile"].iloc[0]
        relative_rows.append(
            {
                "배치": f"B{batch}",
                "하위 수명군 n / 로그 분산 중앙값": f"{int(lower.n)} / {lower.dq_logvar_median:.3f}",
                "상위 수명군 n / 로그 분산 중앙값": f"{int(upper.n)} / {upper.dq_logvar_median:.3f}",
            }
        )
    return f"""### 수명 분포와 장·단수명 비율

{table(pd.DataFrame(distributions), decimals=1)}

동일한 150~2,300회 Histogram으로 비교했다. B1은 500~1,200회대에 분포하고, B2는 사분위 범위 439.5~508.5회에 집중하면서 일부 장수명 셀 때문에 오른쪽 꼬리가 나타난다. B3는 중앙값이 높고 최대 1,935회까지 이어진다. 비율의 분모는 수명 라벨이 유효한 셀이다. B1의 단수명 학습 부족은 B2 과대 예측을 해석하는 근거다.

### 열화 곡선과 Knee 탐색

초기 용량 증가와 이후 감소가 혼재하며, 구간 선형 적합의 후반 기울기 중앙값은 세 배치 모두 전반보다 더 음수여서 후반 가속 열화를 시사한다. B2의 절대 단수명 28셀은 Knee 후보 중앙값 {short.knee.median():.1f}회·후반 기울기 {short.slope_after.median():.2e}Ah/회, 장수명 3셀은 {long.knee.median():.1f}회·{long.slope_after.median():.2e}Ah/회다. 이 표본에서는 단수명군의 급격한 감소가 더 일찍 나타났지만 작은 장수명군만으로 일반 법칙이나 원인을 확정하지 않는다.

{table(pd.DataFrame(knees), decimals=0)}

Knee 표는 수명 결측 셀도 포함한 적합 가능 곡선 기준이며, 수명 비율 표와 분모가 다르다. 평활·구간 적합에 의존하는 탐색적 후보이지 화학적 전이의 확정점이 아니다. 전체 수명에서 계산하므로 모델 X에서 제외했다.

### 초기 ΔQ(V)의 장·단수명 차이

B2 단수명군의 ΔQ 전압 평균(셀별 평균의 중앙값)은 {short.dq_mean.median():.4f}Ah로 장수명군 {long.dq_mean.median():.4f}Ah보다 음의 변화가 크다. 로그 분산 중앙값도 {short.dq_logvar.median():.3f} 대 {long.dq_logvar.median():.3f}로 단수명군이 더 커, 전압에 따른 곡선 변화가 더 크고 덜 평탄한 경향을 요약한다. 분산만으로 모든 전압에서 동일한 형태 차이를 주장하지 않는다.

B1/B3에는 <500회 셀이 없어 절대 장·단수명 비교를 만들지 않았다. 배치 내 Q1 이하/Q3 이상 상대 그룹으로 보완했으며, 세 배치 모두 하위 수명군의 로그 분산 중앙값이 더 컸다.

{table(pd.DataFrame(relative_rows))}

### 충전 프로토콜별 평균 수명

배치별 유효 라벨 정책의 최저·최고 평균 예시다. 전체 정책은 `results/policy_stats.csv`에 있다.

{table(pd.DataFrame(policy_examples), decimals=1)}

C1과 수명 상관은 B1 -0.580, B2 +0.191, B3 -0.083으로 고속 충전이 항상 단수명이라는 결론을 지지하지 않는다. 정책별 작은 n, 전환 SOC·온도·측정 조건의 혼재를 고려해야 하며 정책 평균 차이는 인과효과가 아니다. 이를 근거로 정책 변수는 보조 세트 C에서 검증했다.
"""
