# 성능 진단 리포트

## 실험 및 상태

Seed {{ seed }}, 이미지 클래스 {{ classes }}. 클래스당 Train {{ shots }}장, Validation {{ validation_per_class }}장, Test {{ test_per_class }}장. 입력 시계열은 {{ observations }}개 관측이다.

검수 태그 작성: **{{ reviewed }}/{{ total_errors }}건**. CLI는 미검수 상태에서도 리포트를 생성한다. 검수 통계는 허용된 태그의 작성 건수를 반영하며, 실제 오분류 원인을 모델의 확신도나 예측 결과만으로 확정할 수 없다.

## 이미지 실측 비교

{{ image_metrics_table }}

Validation loss 기준 선택 전략: **{{ selected_strategy }}**. 동일 예산의 scratch 대비 Fine-tuning Test 정확도 차이는 **{{ transfer_gain }}%p**이고, 증강 + weight decay는 기본 Fine-tuning 대비 **{{ augmentation_gain }}%p**다. 이 결과는 한 seed의 사전 지정 실험이며 인과적 개선이나 최적 전략을 보장하지 않는다.

Few-shot에서는 적은 라벨로 시각 특징 전체를 새로 학습하기 어렵다. 전이학습은 ImageNet에서 학습한 경계·질감·형태 표현을 재사용한다. Linear Probing은 backbone을 동결해 학습할 파라미터를 줄이고, Fine-tuning은 전체 표현을 목표 클래스에 맞춘다. 장점은 적은 목표 데이터와 학습 예산으로 얻는 정확도이며, 외부 사전학습 데이터·비용을 포함하므로 Scratch와 총비용이 같은 비교는 아니다.

## 학습 곡선과 편향·분산

{{ loss_diagnosis_table }}

Validation/Train > {{ ratio_threshold }}는 격차를 찾기 위한 휴리스틱이며 확정 진단이 아니다. Train·Validation 손실, 정확도, 베이스라인, 데이터 분포를 함께 해석해야 한다.

High Variance 대응으로 Train에만 flip·밝기/대비·crop 증강을 적용하고 weight decay를 추가한 전략을 비교했다. 증강은 제한된 표본의 위치·조명 변화에 대한 의존을 줄이고, weight decay는 가중치가 과도하게 커지는 것을 억제하려는 선택이다. 학습하는 전략은 Validation loss로 checkpoint를 선택하고 미개선 시 조기 종료해 과도한 학습을 제한한다. Linear Probing의 backbone 동결도 학습 자유도를 줄인다.

위 정확도 차이가 음수이면 이 설정에서 증강·정규화의 일반화 개선은 확인되지 않은 것이다. 두 방법을 동시에 바꿨으므로 효과를 분리할 수 없으며, crop으로 판별 단서가 사라졌는지·변환 강도가 적절한지는 별도 가설이다. 후속 검증은 Validation에서 증강 종류·강도와 weight decay를 하나씩 바꾸고, 학습 곡선과 오류를 함께 비교해야 한다. 이 실험의 증강은 오류 태깅 전에 지정했으며 사후 오류 분석에 따른 조치로 소급 해석하지 않는다.

{{ loss_images }}

## 시계열 실측과 개선

동일 Test 날짜, 동일 원 단위로 비교했다.

{{ series_metrics_table }}

기본 LSTM 대비 LSTM_residual의 MAE 개선률은 **{{ residual_gain }}%**다. Validation MAE 기준 선택 베이스라인은 **{{ selected_baseline }}**이며 Test로 파라미터를 고르지 않았다.

### 베이스라인별 개선률

개선률(%)은 `100 × (베이스라인 오차 − 모델 오차) / 베이스라인 오차`이며, 음수는 악화를 의미한다.

{{ comparison_table }}

기준 오차가 0이거나 지표가 누락되면 개선률은 정의되지 않으므로 N/A로 표시한다.

### 예측 결과와 해석

![예측과 실제](timeseries/predictions.png)

![베이스라인 비교](timeseries/baseline_comparison.png)

시계열 Test 평균은 Train 평균보다 {{ test_train_drift }} KRW/USD 차이가 난다. 이번 평가는 거래일 기준 rolling one-step 예측이며 장기 예측 성능을 의미하지 않는다. RNN MAE는 {{ rnn_mae }}, LSTM MAE는 {{ lstm_mae }}다.

Naive는 마지막 실측값을 그대로 다음 관측의 예측으로 사용한다. 관측 간 수준 변화가 작으면 강한 기준이 되며, 수준을 직접 예측하는 신경망은 제한된 표본과 분포 변화 때문에 이를 넘지 못할 수 있다. 평균 이동은 관찰된 사실이지만 성능 열세의 단독 원인을 입증하지 않는다. 현재 입력은 과거 환율 한 변수뿐이므로 외생 변화의 정보를 직접 제공하지 않는다.

잔차 LSTM은 마지막 관측에 예측 변화량을 더해 수준을 처음부터 복원해야 하는 부담을 줄이는 조치다. 기본 LSTM 대비 개선과 Naive 대비 우위는 별개이므로 [베이스라인별 개선률](#베이스라인별-개선률)을 함께 판단한다. 후속 진단에서는 시간 구간별 오차, 수준·변화량 예측, 모델 크기·입력 길이를 Validation에서 비교한다. 기준 실험의 추가 탐색과 데이터 특성에 근거한 해석은 [기준 실험 보완 분석](../../docs/reference-analysis.md)에 정리했다. 해당 수치는 이 실행에 자동 적용되지 않는다.

## 데이터 특성 및 모델 구조

기본 이미지 입력 크기는 {{ input_size }}×{{ input_size }}이며, 확대만으로 새로운 세부 정보가 생기지는 않는다. 클래스 {{ classes }}의 형태적 유사성과 배경 의존 가능성은 실제 오류 이미지와 클래스별 혼동 통계로 확인해야 한다. 시계열 입력은 {{ window }}개 관측값으로 구성된다.

CNN은 공간상의 국소 패턴을 학습하고, RNN은 순서대로 hidden state를 갱신한다. 일반 RNN은 긴 시퀀스를 역전파할 때 반복되는 미분의 곱으로 기울기가 소실하거나 폭주해 먼 과거의 정보를 학습하기 어렵다. LSTM은 forget/input gate로 cell state의 유지·갱신을 조절하고 output gate로 출력을 조절한다. cell state의 가산적 갱신 경로가 장기 정보와 기울기 전달을 돕지만 소실을 완전히 제거하거나 모든 데이터에서 우위를 보장하지는 않는다. 이번 실측 RNN/LSTM 차이만으로 장기 기억의 우월성을 증명할 수 없다.

ResNet18의 전처리는 RGB → Resize → [0,1] tensor → ImageNet mean/std 정규화다. 기본 입력은 CPU 비용을 줄인 128×128이며 사전학습 가중치의 공식 256 resize/224 crop과 다르다. ResNet의 adaptive pooling으로 실행은 가능하지만 입력 분포가 공식 전처리와 동일하지 않다. ResNet18 전략에 동일한 기본 변환을 적용했으며 128/224 비교 실험은 수행하지 않았다.

### 클래스별 Test 혼동 통계

{{ confusion_table }}

## 오류 진단 및 사람 검수

![실제 Validation 오분류 사례](vision/error_gallery.png)

[오분류 분석표](vision/error_review.csv)는 Validation 오분류를 포함한다. human_tag 작성 건수를 검수 통계에 포함한다.

작성된 실패 원인 태그 통계:

{{ human_tag_counts }}

원본 오류 이미지는 `vision/errors/`에 보관하며, 위 PNG에는 최대 30건을 표시한다. 분석표는 최종 Test 정확도 1위 모델 **{{ error_review_model }}**의 Validation 오류를 대상으로 하므로 전체 이미지의 태그별 오류율이나 다른 모델의 태그별 성능을 나타내지 않는다. 동률이면 실험 목록에서 먼저 나온 모델을 선택한다. 이는 평가 후 오류를 살펴볼 모델의 선택이며 checkpoint 선택은 Validation 기준을 유지한다. 제출 기준의 원인 태깅 최소 30건은 위 검수 건수로 별도 확인해야 하며 CLI 성공 자체가 충족을 보장하지 않는다.

### 실패 패턴과 구조·품질 분리 진단

태그 빈도는 검토한 오류 안에서의 빈도다. 빈도가 높은 패턴을 우선 조사하되 확신도는 원인 증거가 아니며, `model_limitation`도 모델 구조의 결함을 입증하지 않는다. 학습 곡선의 큰 Train/Validation 격차는 과적합 또는 분포 이동 가설을, 저해상도·가림 등의 관찰은 입력 품질 가설을 뒷받침한다. 두 문제는 함께 존재할 수 있다.

구분 절차는 (1) 정답 사례까지 포함한 동일 Validation 표본에 품질 태그를 붙이고, (2) sample_id로 모델별 예측을 맞춰 태그별 오류율을 비교하며, (3) 같은 모델에서 데이터 품질만 바꾼 실험과 같은 데이터에서 모델만 바꾼 실험을 분리하는 것이다. 모델 공통 실패만으로 데이터 문제를 확정하지 않고, 모델별 실패 차이만으로 구조 문제를 확정하지 않는다. 원본에 없는 세부는 단순 확대만으로 복원되지 않는다.

기준 실험에서 확인한 우선 실패 패턴, 학습 곡선과의 교차 해석, 실제 조치와 미실행 후속 검증은 [기준 실험 보완 분석](../../docs/reference-analysis.md)에 기록했다. 새 실행에서는 이 실행의 태그·곡선·지표로 다시 판단해야 한다.

## 재현 및 누수 근거

Look-ahead Bias는 예측 시점에 알 수 없는 미래 관측이나 미래에서 추정한 통계를 학습·입력에 사용하는 오류다. 다음 처리로 평가 구간의 정보가 앞 시점으로 유입되는 것을 막는다.

| 구간 | 코드의 방지 처리 |
| --- | --- |
| 이미지 분할 | 공식 Train에서 Train/Validation 비복원 추출, 공식 Test 별도 사용, 선택 이미지의 분할 간 SHA-256 중복 검사 |
| 시계열 분할 | 시간순 70/10/20 분할, target 날짜로 split 결정 |
| 스케일링 | `timeseries/data.py:prepare_series`에서 `values[:train_end]`의 mean/std만 사용 |
| 윈도우 | 입력은 `[t-window,t)`로 target t와 미래를 제외 |
| 피처 생성 | `timeseries/evaluation.py:baseline_predictions`에서 `shift(1)` 후 rolling/ewm |
| 선택·평가 | Validation으로 checkpoint·baseline 선택, Test는 고정 모델의 rolling one-step 평가 |

Validation/Test 첫 입력에 이전 split의 과거 관측을 포함하고 Test 내부의 과거 실측을 다음 날 입력으로 사용하는 것은 해당 예측 시점에 알려진 정보다. 전체 Test 기간을 한 번에 예측하는 방식과 구별한다. 미래 Test 값 변조에 대한 scaler·과거 입력 불변성은 계약 테스트로 확인한다.

베이스라인 코드는 학습 루프와 별도인 `timeseries/evaluation.py`에 모았다. 학습 없는 인과적 예측과 공통 오차 계산을 독립적으로 검사하고, 모든 모델을 같은 날짜·단위·지표로 비교하기 위한 분리다. `data.py`는 분할·정규화·윈도우, `training.py`는 학습, `pipeline.py`는 실행·저장을 맡는다.

[누수 방지·실행 정책](../../docs/protocol.md), 각 트랙 audit.json, 시계열 predictions.csv, 모델별 history CSV, 이미지 membership.csv 및 [베이스라인 비교](#베이스라인별-개선률)를 함께 확인한다.
