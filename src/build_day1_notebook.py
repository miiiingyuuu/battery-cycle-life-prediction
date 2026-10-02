"""Day1의 질문별 계산·표·그림을 순서대로 실행하는 노트북을 만듭니다."""

from pathlib import Path
import nbformat as nbf

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    cells = []

    def markdown(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text))

    markdown(
        """# ESS 배터리 수명 예측 — Day1 EDA
**울산 1반 · 박민규**

이 노트북은 Q1~Q5의 계산 과정과 결과를 직접 확인하는 실행본입니다. 표·그래프 출력이 저장되어 있어 실행하지 않아도 읽을 수 있습니다.
다시 계산하려면 `mini-project` 또는 `DS` 폴더에서 노트북을 열고 위에서부터 실행하세요. 원본 데이터는 `DS/data/battery-cycle`에 있어야 합니다.
`src/day1_eda.py`와 `src/day1_charge_analysis.py`의 읽기·시각화 함수를 사용합니다. 모델 학습은 Day2에서 수행합니다.

커널에 `requirements-notebook.txt`의 라이브러리를 설치하세요. 원본은 수정하지 않으며, 그림과 통계 파일은 재생성됩니다."""
    )
    code("""from pathlib import Path
import sys
import numpy as np
import pandas as pd
from IPython.display import display, Image

# 실행 위치가 mini-project, DS 또는 notebooks여도 같은 프로젝트를 찾습니다.
project_candidates = [Path.cwd(), Path.cwd() / "mini-project", Path.cwd().parent]
PROJECT_DIR = next(path.resolve() for path in project_candidates if (path / "src/day1_eda.py").exists())
sys.path.insert(0, str(PROJECT_DIR / "src"))
import day1_eda as eda
import day1_charge_analysis as charge
eda.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def show_figure(filename):
    display(Image(filename=str(eda.FIGURES_DIR / filename), width=1000))

print("프로젝트:", PROJECT_DIR)
print("데이터:", eda.DATA_DIR)
""")
    markdown("""## 0. 원본 로드와 분석 단위
원시 139셀을 읽되 모델 입력은 셀당 1행입니다. 수명 결측을 관측 길이로 대체하지 않습니다.
초기 피처는 10~100회에서 계산하고, 전 수명 knee·끝 용량·관측 길이는 설명용입니다.""")
    code(
        """cell_features, capacity_curves, delta_curves, batch_audits = eda.load_batches()
batch_stats = eda.summarize_batches(cell_features)
correlations = eda.calculate_correlations(cell_features)
eda.save_results(cell_features, batch_stats, correlations, batch_audits)
display(pd.DataFrame(batch_audits))
display(cell_features.head())
display(cell_features.groupby("batch").agg(raw_cells=("cell_id", "count"), labeled=("cycle_life", "count")))
display(cell_features.loc[cell_features.cycle_life.isna(), ["cell_id", "policy", "cycle_life"]])"""
    )
    markdown(
        """## Q1. 수명 분포와 단수명 비율
평균·중앙값·IQR·500회 미만 및 1,000회 초과 비율을 배치별로 계산합니다.
B2의 단수명 비율이 높고, B1에서는 550회 기준 소수 클래스가 1셀뿐이므로 회귀를 선택합니다."""
    )
    code(
        """display(batch_stats)
eda.plot_cycle_life(cell_features)
show_figure("01_life.png")
for batch, values in cell_features.groupby("batch"):
    life = values.cycle_life.dropna()
    print(f"B{batch}: <500회={(life < 500).sum()}/{len(life)}, >1000회={(life > 1000).sum()}/{len(life)}, <550회={(life < 550).sum()}/{len(life)}")"""
    )
    markdown(
        """### 단수명 셀과 비교군의 초기 신호
B2 안에서 <500회 / ≥500회를 비교합니다. 표는 중앙값이며 원인 추정이나 새 모델 라벨이 아닙니다."""
    )
    code(
        """b2_labeled = cell_features.query("batch == 2").dropna(subset=["cycle_life"]).copy()
b2_labeled["life_group"] = np.where(b2_labeled.cycle_life < 500, "<500", ">=500")
features = ["cycle_life", "dq_logvar", "Tavg_mean", "chargetime_mean", "chargetime_median", "C1", "I10_positive_mean"]
display(b2_labeled.groupby("life_group")[features].agg(["count", "median", lambda values: values.quantile(.25), lambda values: values.quantile(.75)]))"""
    )
    markdown(
        """## Q2. 용량 열화와 knee 후보
전체 열화 곡선을 보고 연속 구간 선형 모델을 단일 직선과 비교합니다.
탐색적 지지 기준: SSE 감소 ≥20%, ΔBIC >10, 후반 기울기 < 전반 기울기 <0, 평활 변경 위치 폭 ≤ 관측 범위의 5%.
B1 예시 c0는 기준 미충족입니다. 평활 오차는 독립적이지 않아 BIC를 유의성 검정으로 해석하지 않습니다."""
    )
    code("""eda.plot_degradation(cell_features, capacity_curves)
show_figure("02_degradation.png")
eda.plot_knee_examples(cell_features, capacity_curves)
show_figure("08_knee.png")
eda.save_diagnostic_summaries(cell_features)
diagnostics = pd.read_csv(eda.RESULTS_DIR / "diagnostic_summary.csv")
display(diagnostics[["batch", "n", "knee_fit_n", "knee_supported", "sse_gain_median", "bic_gain_median", "knee_median"]])
# 임의 셀의 계산을 직접 재확인합니다.
cycles, capacity = capacity_curves["b1c0"]
display(pd.Series(eda.knee_diagnostics(cycles, capacity), name="b1c0"))""")
    markdown(
        """## Q3. ΔQ100−10의 계산과 통계 피처
실제 전압 축 Vdlin의 각 지점에서 `Qdlin[99] − Qdlin[9]`를 계산합니다.
아래는 한 셀의 ΔQ 배열에서 분산·로그 분산·최소·평균을 다시 계산하여 저장된 피처와 대조하는 과정입니다."""
    )
    code("""voltage, delta_q = delta_curves["b1c0"]
manual_features = {"dq_var": np.var(delta_q), "dq_logvar": np.log10(max(np.var(delta_q), 1e-12)), "dq_min": delta_q.min(), "dq_mean": delta_q.mean()}
saved_features = cell_features.set_index("cell_id").loc["b1c0", list(manual_features)]
display(pd.DataFrame({"직접 계산": manual_features, "저장된 값": saved_features}))
np.testing.assert_allclose(list(manual_features.values()), saved_features.to_numpy(dtype=float))
print("전압 범위:", voltage.min(), voltage.max(), "V / 지점 수:", len(voltage))
eda.plot_delta_curves(cell_features, delta_curves)
show_figure("03_delta.png")
eda.plot_delta_signal(cell_features)
show_figure("04_delta_signal.png")
display(correlations.query("feature in ['dq_logvar', 'dq_min', 'dq_mean']"))""")
    markdown(
        """### 상대 수명 그룹과 전압 민감도
B1·B3에는 <500회 셀이 없어 하위/상위 사분위 그룹도 비교합니다. 상대 그룹은 EDA 표시용입니다.
상수 오프셋을 제거해도 분산은 같지만, 다른 측정 구조의 차이가 해결됐다는 의미는 아닙니다."""
    )
    code(
        """eda.plot_relative_delta(cell_features, delta_curves)
show_figure("09_delta_relative.png")
display(pd.read_csv(eda.RESULTS_DIR / "relative_delta_groups.csv"))
display(diagnostics[["batch", "anchor_max_abs", "centered_max_difference", "full_core_spearman"]])"""
    )
    markdown("""## Q4. 충전 정책·전류 조건과 수명
정책별 n/평균/표준편차를 표시합니다. 표준편차를 계산할 수 없는 n=1을 분산 0으로 해석하지 않습니다.
첫 단계 C-rate와 수명 관계는 배치별 부호가 달라 인과효과로 해석할 수 없습니다.""")
    code(
        """eda.plot_policy_life(cell_features)
show_figure("05_policy.png")
display(pd.read_csv(eda.RESULTS_DIR / "policy_stats.csv"))
eda.plot_charge_rate(cell_features)
show_figure("06_crate.png")
display(correlations.query("feature in ['C1', 'C2', 'I10_positive_mean', 'QD_early_slope']"))"""
    )
    markdown(
        """### 실제 I(t) 패턴을 직접 확인
각 배치의 c0·c14를 고정해 10회 사이클의 전류를 표시합니다. 양수 충전 구간의 단계 전환과 후반 감소를 관찰하세요.
정책의 명목 전류 점선은 실제 SOC가 아닙니다. 표의 초기 용량 기울기와 ΔQ 신호를 함께 보되 6개 예시로 인과성을 주장하지 않습니다."""
    )
    code(
        """patterns, durations = charge.load_charge_records()
charge.plot_current_patterns(patterns, cell_features)
show_figure("10_current_patterns.png")
display(cell_features.loc[cell_features.cell_id.isin(patterns), ["cell_id", "policy", "cycle_life", "QD_early_slope", "dq_logvar"]])"""
    )
    markdown(
        """### 충전시간 극단 기록의 출처
평균 53분대가 나온 셀을 사이클별로 확인합니다. 예: b2c14의 50회, b2c32의 71회에 약 3,934분 기록이 있습니다.
극단값의 원인이 실제 충전·휴지 시간·실험 중단·기록 문제 중 무엇인지는 아직 확인되지 않았습니다.
중앙값을 B의 기본 피처로 두고 평균/제외를 동일 B1 CV에서 비교합니다. 30분 경계는 집계용이며 자동 삭제 기준이 아닙니다."""
    )
    code(
        """charge.plot_charge_time_records(durations)
show_figure("11_charge_time.png")
charge.save_charge_audit(durations, cell_features)
display(pd.read_csv(eda.RESULTS_DIR / "charge_time_audit.csv"))
display(durations.query("chargetime > 1000").sort_values(["batch", "cell_id", "cycle"]))
display(cell_features.loc[cell_features.chargetime_mean > 30, ["cell_id", "chargetime_mean", "chargetime_median", "chargetime_max", "chargetime_gt30_n"]])"""
    )
    markdown("""## Q5. 초기 신호의 상관관계와 다중공선성
Pearson과 Spearman을 함께 확인하고 결측 쌍별 유효 n을 표시합니다. 상관관계는 예측 성능이 아닙니다.
Tavg/Tmax의 중복을 피하고, QD100−QD10은 배치별 부호가 달라 추가 검증 후보로 둡니다.""")
    code("""eda.plot_feature_correlations(cell_features)
show_figure("07_corr.png")
display(correlations)
for batch, values in cell_features.groupby("batch"):
    print(f"B{batch}: Tavg/Tmax r={values.Tavg_mean.corr(values.Tmax_mean):.3f}")""")
    markdown(
        """## 데이터 품질 민감도
B1 원본 46셀 유지가 주 분석이며 끝 용량 >0.885Ah 후보를 자동 삭제하지 않습니다.
전체 수명 품질 플래그는 모델 입력이 아닙니다. 제외 보조 모델도 같은 검증 셀로 비교해야 합니다."""
    )
    code(
        """display(diagnostics[["batch", "raw_label_n", "unflagged_label_n", "raw_dq_r", "unflagged_dq_r"]])
display(cell_features.groupby("batch")[["bad_QD_rows", "endpoint_quality_flag"]].sum())"""
    )
    markdown(
        """## EDA → 모델 전략
- 타깃: 초기 100회로 총 `cycle_life`를 예측하는 회귀. 성능은 아직 미측정.
- A: dq_logvar.
- B: A + QD_early_slope + Tavg_mean + chargetime_median.
- C: B + C1 + switch_pct + C2.
- 1차: 중앙값 Dummy, A 선형 회귀, B/C Ridge. alpha={0.1,1,10}, 원/로그 타깃.
- 추가: B+QD100_minus_QD10 및 기울기 대체 Ridge; B의 충전시간을 평균/제외로 바꾼 Ridge. 같은 CV와 1-SE 규칙 적용.
- B1 정책 그룹 hold-out(seed=42)과 개발 데이터 GroupKFold(5). 전처리는 fold train에서만 fit.
- 모든 비교를 hold-out 확인 전에 확정. 선택 고정 후 전체 B1로 재학습하고 B2·B3 평가.
- B2·B3 라벨을 EDA에서 봤으므로 완전한 blind test로 주장하지 않음.

전류·충전시간의 품질 확인은 도메인 관찰이며 B2 예측 성능으로 규칙을 조정하지 않습니다."""
    )
    code("""feature_sets = {
    "A": ["dq_logvar"],
    "B": ["dq_logvar", "QD_early_slope", "Tavg_mean", "chargetime_median"],
    "C": ["dq_logvar", "QD_early_slope", "Tavg_mean", "chargetime_median", "C1", "switch_pct", "C2"],
}
for name, features in feature_sets.items():
    print(name, features)
    display(cell_features.groupby("batch")[features].count())
assert diagnostics.knee_fit_n.tolist() == [46, 45, 46]
assert diagnostics.knee_supported.tolist() == [45, 44, 46]
print("EDA 계산 및 주요 숫자 검증 완료. 모델 학습은 수행하지 않았습니다.")""")
    # 보완 분석을 관련 질문 바로 뒤에 배치합니다.
    additions = [
        (
            "## Q2.",
            "### 배치별 최단 셀과 통계적 이상치 구분\n<500회 셀과 IQR 하한 미만의 통계적 이상치는 다른 정의입니다. 각 배치 최단 셀의 정책·초기 신호를 확인합니다.",
            """for batch, values in cell_features.groupby('batch'):
    labeled = values.dropna(subset=['cycle_life'])
    q1, q3 = labeled.cycle_life.quantile([.25, .75])
    lower = q1 - 1.5*(q3-q1)
    print(f'B{batch}: IQR 하한 {lower:.1f}, 하한 미만 {(labeled.cycle_life < lower).sum()}셀')
    display(labeled.nsmallest(3, 'cycle_life')[['cell_id', 'policy', 'cycle_life', 'dq_logvar', 'Tavg_mean', 'C1']])""",
        ),
        (
            "## Q5.",
            "### 전류 패턴 요약과 초기 용량 변화율의 직접 상관\n기울기 <0은 초기 용량 감소, >0은 증가입니다. 수명 결측 셀도 두 초기 피처가 있으면 포함합니다.",
            """charge.plot_current_degradation(cell_features)
show_figure('12_current_degradation.png')
rows = []
for batch, values in cell_features.groupby('batch'):
    pairs = values[['I10_positive_mean', 'QD_early_slope']].dropna()
    rows.append({'batch':batch, 'n':len(pairs), 'Pearson':pairs.I10_positive_mean.corr(pairs.QD_early_slope), 'Spearman':pairs.I10_positive_mean.corr(pairs.QD_early_slope, method='spearman')})
display(pd.DataFrame(rows))""",
        ),
        (
            "## 데이터 품질",
            "### 가장 강한 수명 관계와 피처 중복\n민감도 변형을 제외한 초기 피처에서 절대 Pearson 상관을 비교합니다. B2의 충전시간 중앙값은 ΔQ보다 강하지만 부호가 배치별로 바뀝니다. 가장 강한 상관과 최적 예측 모델은 다릅니다.",
            """candidates = correlations.query("feature not in ['dq_core_logvar', 'dq_centered_logvar']").copy()
candidates['abs_r'] = candidates.pearson.abs()
display(candidates.sort_values(['batch', 'abs_r'], ascending=[True, False]).groupby('batch').head(3))
for batch, values in cell_features.groupby('batch'):
    print(f'B{batch} 초기 피처 간 Pearson 상관')
    display(values[['dq_logvar', 'dq_min', 'dq_mean', 'QD_early_slope', 'QD100_minus_QD10', 'Tavg_mean', 'Tmax_mean', 'chargetime_median']].corr())""",
        ),
    ]
    for prefix, text, source in additions:
        index = next(
            i
            for i, cell in enumerate(cells)
            if cell.cell_type == "markdown" and cell.source.startswith(prefix)
        )
        cells[index:index] = [
            nbf.v4.new_markdown_cell(text),
            nbf.v4.new_code_cell(source),
        ]

    notebook = nbf.v4.new_notebook(
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
    path = PROJECT_DIR / "DAY1-EDA-울산_1반-박민규.ipynb"
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
