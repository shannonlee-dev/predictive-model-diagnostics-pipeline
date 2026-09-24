# 성능 진단 리포트

## 실험 및 상태

Seed {{ seed }}, 이미지 클래스 {{ classes }}. 클래스당 Train {{ shots }}장, Validation {{ validation_per_class }}장, Test {{ test_per_class }}장. 입력 시계열은 {{ observations }}개 관측이다.

사람 검수: **{{ reviewed }}/{{ total_errors }}건**. CLI는 미검수 상태에서도 리포트를 생성한다. 검수 통계는 사람이 확인해 작성한 태그만 반영하며, 실제 오분류 원인을 모델의 확신도나 예측 결과만으로 확정할 수 없다.

## 이미지 실측 비교

{{ image_metrics_table }}

Validation loss 기준 선택 전략: **{{ selected_strategy }}**. 동일 예산의 scratch 대비 Fine-tuning Test 정확도 차이는 **{{ transfer_gain }}%p**이고, 증강 + weight decay는 기본 Fine-tuning 대비 **{{ augmentation_gain }}%p**다. 이 결과는 한 seed의 사전 지정 실험이며 인과적 개선이나 최적 전략을 보장하지 않는다.

## 학습 곡선과 편향·분산

{{ loss_diagnosis_table }}

Validation/Train > {{ ratio_threshold }}는 격차를 찾기 위한 휴리스틱이며 확정 진단이 아니다. Train·Validation 손실, 정확도, 베이스라인, 데이터 분포를 함께 해석해야 한다.

{{ loss_images }}

## 시계열 실측과 개선

{{ series_metrics_table }}

기본 LSTM 대비 LSTM_residual의 MAE 개선률은 **{{ residual_gain }}%**다. Validation MAE 기준 선택 베이스라인은 **{{ selected_baseline }}**이며 Test로 파라미터를 고르지 않았다.

![예측과 실제](timeseries/predictions.png)

![베이스라인 비교](timeseries/baseline_comparison.png)

시계열 Test 평균은 Train 평균보다 {{ test_train_drift }} KRW/USD 차이가 난다. 이번 평가는 거래일 기준 rolling one-step 예측이며 장기 예측 성능을 의미하지 않는다. RNN MAE는 {{ rnn_mae }}, LSTM MAE는 {{ lstm_mae }}다.

## 데이터 특성 및 모델 구조

이미지 입력 크기는 {{ input_size }}×{{ input_size }}이며, 확대만으로 새로운 세부 정보가 생기지는 않는다. 클래스 {{ classes }}의 형태적 유사성과 배경 의존 가능성은 실제 오류 이미지와 클래스별 혼동 통계로 확인해야 한다. 시계열 입력은 {{ window }}개 관측값으로 구성된다.

CNN은 공간상의 국소 패턴을 학습하고, RNN은 순서대로 hidden state를 갱신한다. LSTM은 gate와 cell state를 통해 장기 정보 경로를 추가하지만, 이번 결과만으로 장기 기억의 우월성을 증명할 수 없다.

### 클래스별 Test 혼동 통계

{{ confusion_table }}

## 오류 진단 및 사람 검수

![실제 Validation 오분류 사례](vision/error_gallery.png)

[오분류 분석표](vision/error_review.csv)는 Validation 오분류를 포함한다. 실제 관찰에 따른 human_tag만 사람 검수 통계에 포함한다.

사람이 확인한 실패 원인 통계:

{{ human_tag_counts }}

원본 오류 이미지는 `vision/errors/`에 보관하며, 위 PNG에는 최대 30건을 표시한다.

## 재현 및 누수 근거

[누수 방지·실행 정책](../../docs/protocol.md), 각 트랙 audit.json, predictions.csv, history.csv, membership.csv 및 [베이스라인 비교](baseline_comparison.md)를 함께 확인한다.
