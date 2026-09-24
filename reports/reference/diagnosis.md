# 성능 진단 리포트

## 실험 및 상태

Seed 42, 이미지 클래스 cat, deer, dog. 클래스당 Train 40장, Validation 150장, Test 250장. 입력 시계열은 1249개 관측이다.

## 이미지 실측 비교

| model | split | loss | accuracy |
| --- | --- | --- | --- |
| scratch | Train | 0.6364 | 0.8167 |
| scratch | Validation | 1.0692 | 0.4467 |
| scratch | Test | 1.0967 | 0.4213 |
| linear_probe | Train | 0.1947 | 0.9667 |
| linear_probe | Validation | 0.5269 | 0.7822 |
| linear_probe | Test | 0.5458 | 0.7680 |
| fine_tune | Train | 0.0021 | 1.0000 |
| fine_tune | Validation | 0.6695 | 0.7911 |
| fine_tune | Test | 0.6914 | 0.7533 |
| augmented | Train | 0.1042 | 0.9750 |
| augmented | Validation | 0.7790 | 0.7267 |
| augmented | Test | 0.7569 | 0.7147 |

Validation loss 기준 선택 전략: **linear_probe**. 동일 예산의 scratch 대비 Fine-tuning Test 정확도 차이는 **33.20%p**이고, 증강 + weight decay는 기본 Fine-tuning 대비 **-3.87%p**다. 이 결과는 한 seed의 사전 지정 실험이며 인과적 개선이나 최적 전략을 보장하지 않는다.

## 학습 곡선과 편향·분산

| model | best_epoch | Train_loss | Validation_loss | gap | interpretation |
| --- | --- | --- | --- | --- | --- |
| scratch | 3 | 0.6364 | 1.0692 | 0.4328 | High Variance 또는 분포 이동 의심 |
| linear_probe | 10 | 0.1947 | 0.5269 | 0.3321 | High Variance 또는 분포 이동 의심 |
| fine_tune | 6 | 0.0021 | 0.6695 | 0.6674 | High Variance 또는 분포 이동 의심 |
| augmented | 2 | 0.1042 | 0.7790 | 0.6748 | High Variance 또는 분포 이동 의심 |
| RNN | 49 | 0.0114 | 0.0136 | 0.0022 | 큰 일반화 격차 없음; 절대 오차·베이스라인과 함께 편향 판단 |
| LSTM | 50 | 0.0170 | 0.0162 | -0.0008 | 큰 일반화 격차 없음; 절대 오차·베이스라인과 함께 편향 판단 |
| LSTM_residual | 1 | 0.0074 | 0.0097 | 0.0023 | 큰 일반화 격차 없음; 절대 오차·베이스라인과 함께 편향 판단 |

Validation/Train > 1.5는 격차를 찾기 위한 휴리스틱이며 확정 진단이 아니다. Train·Validation 손실, 정확도, 베이스라인, 데이터 분포를 함께 해석해야 한다.

![scratch Train/Validation loss](vision/scratch_loss.png)
![linear_probe Train/Validation loss](vision/linear_probe_loss.png)
![fine_tune Train/Validation loss](vision/fine_tune_loss.png)
![augmented Train/Validation loss](vision/augmented_loss.png)
![RNN Train/Validation loss](timeseries/RNN_loss.png)
![LSTM Train/Validation loss](timeseries/LSTM_loss.png)
![LSTM_residual Train/Validation loss](timeseries/LSTM_residual_loss.png)

## 시계열 실측과 개선

| model | MAE | RMSE | MAPE | MAPE_excluded |
| --- | --- | --- | --- | --- |
| Naive | 5.0737 | 6.8293 | 0.3720 | 0 |
| SMA5 | 7.4883 | 9.6573 | 0.5486 | 0 |
| SMA10 | 9.5579 | 12.3129 | 0.6990 | 0 |
| SMA20 | 13.7652 | 17.3965 | 1.0037 | 0 |
| EMA0.1 | 11.9989 | 15.1703 | 0.8746 | 0 |
| EMA0.3 | 6.9378 | 9.1102 | 0.5078 | 0 |
| EMA0.5 | 5.7385 | 7.6743 | 0.4206 | 0 |
| RNN | 7.9137 | 11.1535 | 0.5726 | 0 |
| LSTM | 8.0968 | 11.1204 | 0.5881 | 0 |
| LSTM_residual | 5.0918 | 6.8393 | 0.3733 | 0 |

기본 LSTM 대비 LSTM_residual의 MAE 개선률은 **37.11%**다. Validation MAE 기준 선택 베이스라인은 **Naive**이며 Test로 파라미터를 고르지 않았다.

![예측과 실제](timeseries/predictions.png)

![베이스라인 비교](timeseries/baseline_comparison.png)

시계열 Test 평균은 Train 평균보다 144.95 KRW/USD 차이가 난다. 이번 평가는 거래일 기준 rolling one-step 예측이며 장기 예측 성능을 의미하지 않는다. RNN MAE는 7.9137, LSTM MAE는 8.0968다.

## 데이터 특성 및 모델 구조

이미지 입력 크기는 128×128이며, 확대만으로 새로운 세부 정보가 생기지는 않는다. 클래스 cat, deer, dog의 형태적 유사성과 배경 의존 가능성은 실제 오류 이미지와 클래스별 혼동 통계로 확인해야 한다. 시계열 입력은 30개 관측값으로 구성된다.

CNN은 공간상의 국소 패턴을 학습하고, RNN은 순서대로 hidden state를 갱신한다. LSTM은 gate와 cell state를 통해 장기 정보 경로를 추가하지만, 이번 결과만으로 장기 기억의 우월성을 증명할 수 없다.

### 클래스별 Test 혼동 통계

| model | actual | predicted | count |
| --- | --- | --- | --- |
| scratch | cat | cat | 63 |
| scratch | cat | deer | 77 |
| scratch | cat | dog | 110 |
| scratch | deer | cat | 39 |
| scratch | deer | deer | 137 |
| scratch | deer | dog | 74 |
| scratch | dog | cat | 64 |
| scratch | dog | deer | 70 |
| scratch | dog | dog | 116 |
| linear_probe | cat | cat | 181 |
| linear_probe | cat | deer | 19 |
| linear_probe | cat | dog | 50 |
| linear_probe | deer | cat | 26 |
| linear_probe | deer | deer | 205 |
| linear_probe | deer | dog | 19 |
| linear_probe | dog | cat | 44 |
| linear_probe | dog | deer | 16 |
| linear_probe | dog | dog | 190 |
| fine_tune | cat | cat | 174 |
| fine_tune | cat | deer | 25 |
| fine_tune | cat | dog | 51 |
| fine_tune | deer | cat | 31 |
| fine_tune | deer | deer | 199 |
| fine_tune | deer | dog | 20 |
| fine_tune | dog | cat | 44 |
| fine_tune | dog | deer | 14 |
| fine_tune | dog | dog | 192 |
| augmented | cat | cat | 178 |
| augmented | cat | deer | 20 |
| augmented | cat | dog | 52 |
| augmented | deer | cat | 31 |
| augmented | deer | deer | 181 |
| augmented | deer | dog | 38 |
| augmented | dog | cat | 59 |
| augmented | dog | deer | 14 |
| augmented | dog | dog | 177 |

## 오류 진단 및 사람 검수

![실제 Validation 오분류 사례](vision/error_gallery.png)

[오분류 분석표](vision/error_review.csv)는 Validation 오분류를 포함한다. 실제 관찰에 따른 human_tag만 사람 검수 통계에 포함한다.

사람이 확인한 실패 원인 통계:

| human_tag | count |
| --- | --- |
| high_confidence_error | 47 |
| class_similarity | 40 |
| dark_lighting | 7 |

원본 오류 이미지는 `vision/errors/`에 보관하며, 위 PNG에는 최대 30건을 표시한다.

## 재현 및 누수 근거

[누수 방지·실행 정책](../../docs/protocol.md), 각 트랙 audit.json, predictions.csv, history.csv, membership.csv 및 [베이스라인 비교](baseline_comparison.md)를 함께 확인한다.
