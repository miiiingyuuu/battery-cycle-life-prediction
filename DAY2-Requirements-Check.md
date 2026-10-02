# Day2 요구사항 최종 대조

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
