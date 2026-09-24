# 기준 실험: 실패 원인과 대응 근거

이 문서는 `reports/reference/`의 저장된 실험과 현재 오류 태그를 해석한다. 새 실행에 그대로 적용하는 자동 진단이 아니다. 기존 네 학습 전략의 실측을 비교하며, 아래의 조치·결과와 후속 검증을 구분한다. 원자료는 [오류 분석표](../reports/reference/vision/error_review.csv), [이미지 지표](../reports/reference/vision/metrics.csv), [시계열 지표](../reports/reference/timeseries/metrics.csv)다.

## 현재 최고 성능 모델과 오류 자료

현재 모델별 성능은 [진단 리포트의 이미지 실측 비교](../reports/reference/diagnosis.md#이미지-실측-비교), 검수 건수와 태그 집계는 [오류 진단 및 사람 검수](../reports/reference/diagnosis.md#오류-진단-및-사람-검수)에서 확인한다. 아래에는 오류 자료의 복원 경위와 해석 기준을 기록한다.

`final_performance.png`, `error_review.csv`, `error_gallery.png`는 Linear Probing 결과를 담고 있다. 저장된 checkpoint가 없어 Linear Probing을 기존 seed 42·10 epoch로 재현했으며, 기존 Test 예측 클래스 전체와 loss history·confidence가 일치함을 확인한 뒤 Validation 오류를 복원했다. 기존 지표와 Test 예측은 유지했다.

현재 오류 CSV의 고유 이미지를 확인하고 `human_tag`와 `review_note`를 작성했다. 다른 모델에서 가져온 관찰은 이미지 품질에 해당하는 메모를 우선 재사용하고, 예측 클래스에 관한 해석은 현재 예측과 대조했다. 태그는 사용자가 검수·수정할 수 있는 가설이다.

`low_resolution`은 32×32라는 크기 자체가 아니라 해당 이미지에서 구별 단서를 읽기 어려운 경우에 사용한다. `model_limitation`은 형태 단서가 남아 있어도 오류 원인을 특정하지 못한 잠정 분류이며 구조 결함의 증거가 아니다. 태그 빈도는 오류 표본 안에서의 분포이며 전체 Validation의 태그별 오류율이 아니다.

Test 결과는 사후 오류 분석 대상 모델을 선택하는 데만 사용했다. 학습 checkpoint는 계속 Validation loss로 선택한다.

## 1. 이전 Fine-tuning 오류 분석과 실제 조치

아래 1·2절의 Fine-tuning 태그 수치는 이전 분석 기록이다. 원자료는 [이전 Fine-tuning 오류 분석표](../reports/reference/vision/fine_tune_error_review.csv)에 보존했고, 현재 최고 모델의 오류 자료와 구분한다. Fine-tuning Validation 오분류 94개는 고유 sample_id이며 모두 human_tag와 review_note가 작성되어 있다. 원인 태깅 최소 30건을 충족한다. 기록된 검토 내용을 집계했으며 CLI가 실제 사람의 검토 이력을 인증하는 것은 아니다.

| 관찰 태그 | 건수 | 검토 오류 중 비중 |
| --- | ---: | ---: |
| low_resolution | 33 | 35.11% |
| model_limitation | 27 | 28.72% |
| class_similarity | 24 | 25.53% |
| background_clutter | 9 | 9.57% |
| occlusion | 1 | 1.06% |

이전 Fine-tuning 분석에서 우선 조사한 실패 패턴은 **낮은 해상도에서 클래스 구별 단서가 부족한 사례**다. 관찰 태그 중 가장 많고, 단순한 32→128 확대는 원본에 없는 세부를 추가하지 못하기 때문이다. 해당 33건의 평균 확신도는 0.8144로, 확신도가 높아도 정답을 보장하지 않는다. 이는 실패 원인의 확정이나 전체 데이터의 저해상도 오류율이 아니다. 오분류만 검토했으므로 정답 사례를 포함한 분모가 없다. 모든 사례의 비용을 같게 보는 우선순위이며 업무별 피해 규모를 측정한 것은 아니다.

`model_limitation` 27건은 관찰만으로 뚜렷한 품질 원인을 특정하지 못한 사례의 가설로 해석한다. 구조 결함 27건으로 확정하지 않는다. 검토 메모의 불확실성을 유지한다.

| 구분 | 실제 내용 | 결과·한계 |
| --- | --- | --- |
| 사전 지정 대응 | flip·밝기/대비·crop과 weight decay 0.01을 함께 적용 | Test 정확도 75.33→71.47%, −3.87%p. 개선 실패이며 태깅 결과를 보고 설계한 조치가 아님 |
| 기존 모델 선택 | Validation loss 최소인 Linear Probing 선택 | loss 0.5269로 Fine-tuning 0.6695보다 낮음. Test 정확도 76.80%는 사후 평가이며 선택 기준이 아님 |
| 이번 분석 후 조치 | 현재 CSV로 오래된 태그 집계·갤러리·보고서 갱신, 저해상도 우선 조사와 원인 미확정 사례 구분 | 산출물 정합성과 진단 설명 보완. 분류 성능을 개선했다고 주장하지 않음 |
| 후속 검증 — 미실행 | 정답을 포함한 Validation 품질 태깅, 약한 증강/종류별 제거/weight decay 개별 비교, 가능하면 실제 고해상도 원본 확보 | Validation으로 효과를 확인한 뒤 설정 고정. 이미 관찰한 Test 대신 새로운 holdout으로 최종 확인 |

오류 분석 후 원인을 겨냥해 다시 학습한 실험은 없다. 따라서 “오류 분석으로 발견한 저해상도 문제를 해결했고 정확도가 올랐다”는 답변은 할 수 없다. 현재 제출물에서 설명 가능한 것은 발견·우선순위·실제로 적용한 기존 대응·실패 결과·후속 검증 설계다.

## 2. 학습 곡선과 오류를 함께 읽기

Fine-tuning은 선택 checkpoint에서 Train loss 0.0021, Validation loss 0.6695로 격차가 크다. Linear Probing은 0.1947/0.5269다. backbone까지 학습하는 설정의 과적합 가능성을 의심할 근거이며, 모델을 동결하는 선택은 적은 학습 표본에서 자유도를 줄인다. 증강 모델은 0.1042/0.7790이고 Test 정확도도 낮으므로, 이번 강도·조합의 증강이 일반화를 개선했다는 증거는 없다. 증강과 weight decay를 동시에 변경했으므로 어느 쪽이 악화 원인인지 분리할 수 없다.

품질 태그는 저해상도·배경·가림 등 입력 단서의 한계를 보여준다. 학습 격차와 함께 존재하므로 성능 저하를 전부 모델 구조 또는 전부 데이터 품질 탓으로 돌릴 수 없다.

동일한 Test sample_id 750개에 대해 저장된 예측을 맞추면 다음을 얻는다.

| 비교 | 사례 수 |
| --- | ---: |
| Scratch 포함 네 모델 모두 오분류 | 58 |
| 세 전이학습 모델 모두 오분류 | 77 |
| Fine-tuning 오분류, Linear Probing 정분류 | 82 |
| Fine-tuning 정분류, Linear Probing 오분류 | 71 |
| Fine-tuning 전체 오분류 | 185 |
| Fine-tuning cat↔dog 혼동 | 95 |

cat↔dog는 Fine-tuning 오류 185건 중 95건(51.35%)이다. 클래스 구별의 어려움과 일치하는 관찰이지만 형태 유사성이나 배경 사용을 직접 증명하지 않는다. 모델 공통 실패는 품질·라벨·공통 사전학습 표현의 한계를 모두 의심하게 한다. 반대로 모델에 따라 정오가 달라지는 사례는 모든 실패가 입력 정보 부족으로 불가피한 것은 아님을 보여준다.

**Validation 품질 태그와 Test 예측은 다른 표본이다.** 위 77건이 저해상도 33건과 겹친다고 주장할 수 없다. 네 전략 모두 ResNet18이므로 이 비교는 학습 전략 차이의 근거이며 CNN 구조 간 우열의 근거도 아니다.

원인을 더 분리하려면 동일 Validation 이미지의 정답·오답 모두에 품질 태그를 붙여 태그별 오류율을 계산한다. 동일 데이터에서 학습 전략 또는 구조만 바꾸는 실험과 동일 모델에서 품질만 바꾸는 실험을 각각 수행한다. 태그별 공통 실패, 손실 격차, 실제 품질 개입 전후 변화가 함께 일치할 때 원인 가설을 좁힐 수 있다.

## 3. 전이학습 결과의 해석 범위

[이미지 실측 비교](../reports/reference/diagnosis.md#이미지-실측-비교)에서 관측한 전이학습의 이점은 적은 목표 라벨과 동일 최대 epoch 예산에서 높은 정확도다. 학습시간이나 전체 사전학습 비용 우위를 측정한 것은 아니다. 전처리와 학습 조건은 [비교 조건](protocol.md#비교-조건)을 따른다. 공식 224 입력과의 비교가 없으므로 128 입력 사용이 정확도에 미친 영향은 단정하지 않는다.

## 4. 시계열 열세와 데이터 특성

동봉 환율 1,249개 관측의 pandas 표본 표준편차(ddof=1)는 수준 93.3977, 1차 차분 7.1332 KRW/USD이며 수준의 lag-1 상관은 0.9971이다. 이는 **전체 자료의 사후 기술통계**로 scaler fitting이나 모델 선택에 사용하지 않았다. 추세·비정상성도 높은 수준 상관을 만들 수 있으므로 랜덤워크를 입증하는 통계는 아니다. 다만 인접 관측의 수준이 비슷해 직전 실측을 유지하는 Naive가 강한 기준이 될 수 있다는 해석과 일치한다.

| 관찰 | 가능한 해석 | 확인한 대응·결과 |
| --- | --- | --- |
| 기본 LSTM MAE 8.0968, Naive 5.0737 | 수준 직접 예측의 편향, 제한된 학습 자료, 분포 이동 가능성 | 마지막 관측에 변화량을 더하는 잔차 구조에서 MAE 5.0918, 기본 LSTM 대비 37.11% 개선 |
| 잔차 LSTM도 Naive 대비 −0.3573% | 과거 한 변수만으로 학습한 변화량의 추가 예측력이 작거나 불안정할 가능성 | Naive 우위를 넘었다고 결론 내리지 않음 |
| Test 평균이 Train보다 144.95 높음 | 수준 이동에 대한 일반화 어려움 가능성 | 평균 차이만으로 성능 열세의 원인을 확정하지 않음 |
| RNN/LSTM Train·Validation loss의 큰 격차는 없음 | 단순 High Variance 설명만으로 부족 | 원 단위 오차·Naive 비교·시기별 오차와 함께 해석 |

[추가 탐색](../reports/residual-search/report.md)은 입력 길이·hidden 크기·학습률·weight decay·배치·손실함수 3,600개 조합을 검토했다. Validation으로 고정한 설정은 Test 10개 seed에서 승리 0·동률 2·패배 8회, 평균 MAE 5.11044로 Naive 대비 약 0.72% 악화했다. 시기별로 두 구간은 개선, 두 구간은 악화했고 전체 우위를 확보하지 못해 탐색을 종료했다. epoch 0의 Naive 동일 예측이 선택된 동률은 학습 성과로 세지 않는다.

후속 가설은 예측 시점에 실제로 이용 가능한 외생 변수의 추가, 변화량 예측, 더 최근 구간에 맞춘 rolling 학습 등이다. 이 방법의 개선 효과는 미검증이다. 발표·공표 지연을 고려해 피처를 정렬하고 각 fold의 Train만으로 정규화해야 한다. 기존 Test를 이미 관찰했으므로 반복 튜닝 후 같은 구간을 새로운 holdout으로 제시하지 않는다.

## 계산 재현

프로젝트 루트에서 다음 코드는 원자료만 읽는다. 추가 분석 수치를 갱신할 때 이 문서도 함께 검토한다.

```python
from pathlib import Path
import pandas as pd

root = Path('reports/reference/vision')
review = pd.read_csv(root / 'fine_tune_error_review.csv')
print(review.groupby('human_tag').agg(
    count=('sample_id', 'size'), mean_confidence=('confidence', 'mean')))
models = ['scratch', 'linear_probe', 'fine_tune', 'augmented']
predictions = {
    model: pd.read_csv(root / f'{model}_Test_predictions.csv').set_index('sample_id')
    for model in models
}
first = predictions['scratch']
for frame in predictions.values():
    assert frame.index.is_unique and set(frame.index) == set(first.index)
    assert frame.actual.reindex(first.index).equals(first.actual)
wrong = pd.DataFrame({m: f.actual.ne(f.predicted) for m, f in predictions.items()})
print('all models:', wrong.all(axis=1).sum())
print('all transfer:', wrong[models[1:]].all(axis=1).sum())
print('fine wrong / probe right:', (wrong.fine_tune & ~wrong.linear_probe).sum())
print('fine right / probe wrong:', (~wrong.fine_tune & wrong.linear_probe).sum())
fine = predictions['fine_tune']
print('fine errors:', wrong.fine_tune.sum())
print('cat/dog:', (((fine.actual == 0) & (fine.predicted == 2)) |
                   ((fine.actual == 2) & (fine.predicted == 0))).sum())
series = pd.read_csv('datasets/krw_2020_2024.csv').value
print('level std / diff std / lag1:', series.std(), series.diff().std(), series.autocorr())
```
