# 데이터와 재현 경로

원본은 [Kaggle 데이터셋](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)의 version 1이며, [논문 저자 저장소](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)와 처리 조건이 완전히 같지는 않습니다.

Day2는 저장된 `results/cell_features.csv`를 입력으로 사용하므로 원본 MAT 파일 없이 재실행할 수 있습니다. 이 파일은 Day1 코드에서 만든 139셀의 파생 피처·라벨·감사 정보이며 `results/audit.json`에 원본 파일 정보를 기록했습니다. Day2 실행 시 SHA-256으로 입력을 기록합니다.

원본부터 EDA를 재실행하려면 아래 파일을 `data/battery-cycle/`에 넣습니다. 원본 약 7.8 GiB는 Git 저장소와 제출 패키지에 포함하지 않습니다. 데이터 이용 조건은 원본 배포 페이지를 확인하세요.

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat`

기존 실습 폴더의 상위 `../data/battery-cycle/`도 지원합니다. 다른 위치는 `BATTERY_DATA_DIR` 환경변수로 지정할 수 있습니다.

```bash
export BATTERY_DATA_DIR=/absolute/path/to/battery-cycle
python src/day1_eda.py
python src/day1_charge_analysis.py
```

정답 결측은 관측 길이로 대체하지 않습니다. B1 46셀, B2 39셀, B3 44셀만 성능 평가에 사용합니다. 나머지 10셀은 `cycle_life`가 결측입니다. `cell_features.csv`에는 전체 수명 설명용 컬럼도 있으므로 X는 Day2 코드의 허용 피처 목록으로만 선택합니다.
