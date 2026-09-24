# 실험·누수 방지·실패 대응 정책

## 비교 조건

단일 CPU, seed 42, ResNet18, 입력 128×128, 동일 클래스·샘플·batch size 24·최대 10 epoch를 공유한다. Validation cross entropy로 checkpoint를 선택하고 4 epoch 미개선 시 조기 종료한다. Scratch는 무작위 초기화, Linear Probing은 ImageNet 사전학습 backbone 전체 동결(파라미터와 BatchNorm running statistics 포함), Fine-tuning은 전체 학습이다. Linear Probing AdamW learning rate는 0.001, 나머지는 0.0003으로 사전에 정했다. 따라서 동일 learning rate 단일요인 실험은 아니며 학습률의 영향은 분리하지 못한다. 기본 weight decay는 0, 증강 전략은 0.01이다.

Train/Validation/Test의 기본 전처리: RGB, bilinear Resize(128,128), tensor [0,1], ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225]. CPU 비용을 줄이려고 공식 224 crop 대신 128 입력을 쓴다. 정규화 상수는 현재 Validation/Test에서 추정하지 않는다. 증강 실험의 Train에만 flip(p=0.5), brightness=0.3, contrast=0.2, reflect padding=12 후 128 crop을 적용한다. Train loss 측정과 모든 추론에는 증강이 없다.

이미지 Test는 CIFAR-10 공식 Test에서 고정 seed로 샘플링한다. Train과 Validation은 공식 Train에서 클래스별 비복원 추출하며 학습은 클래스당 40장이다. 전체 원본 32×32의 hash로 선택된 분할 간 완전 중복도 검사한다. 근접 중복 및 ImageNet 사전학습 데이터와의 중복까지 입증한 것은 아니다.

## 시계열 누수 방지

| 단계 | 정책 | 검증 근거 |
| --- | --- | --- |
| 입력 | date,value 및 선택적 단일 ticker, 중복 날짜·비유한값 거부 | load_series 및 계약 테스트 |
| 결측 | 연준 휴일 결측 제거, 보간·역방향 채움 없음 | datasets/provenance.json |
| 분할 | 관측 행 기준 Train 70%, Validation 10%, Test 20%, 시간순 | audit.json의 날짜·개수 |
| 특성 | 전체 시계열에서 shift(1) 후 rolling/ewm; 이후 분할 | 미래 값 변조 테스트 |
| 정규화 | 원시 Train prefix에서만 mean/std 산출 | Test 값 변조 시 scaler 불변 테스트 |
| 윈도우 | target t에 [t-window,t), target은 split에 소속 | 미래 값 변경 시 과거 입력 불변 테스트 |
| 경계 문맥 | Validation/Test 초기에 직전 split의 과거 관측 허용 | 예측 시점에 이미 알려진 값만 사용 |
| 반복 평가 | rolling one-step; Test 내 과거 실측은 다음 날짜에 사용 가능 | 모든 모델 동일 target 날짜 |
| 샘플 상태 | 각 window의 RNN/LSTM hidden state를 0에서 시작 | batch의 다른 샘플 변조 불변 테스트 |
| 파라미터 | checkpoint 및 baseline 선택에 Validation만 사용 | best loss/Validation MAE 기록 |
| 개선 | 증강·잔차 구조는 실행 전에 지정, Test 보고 후 재튜닝 없음 | 고정 실험 목록 |

Train 전체에서 scaler를 fit하고 Train 내 이전 날짜의 window를 변환하는 것은 고정 훈련 데이터 학습 방식이다. expanding-window로 과거 시점별 모델을 재학습한 백테스트가 아니다. 윈도우 길이 30의 Train 첫 30개 target은 평가에서 제외되고 Validation/Test 전체는 동일 날짜로 비교한다. 시계열 shuffle=False. 결측 제거 후 30-step은 30개의 관측 거래일이며 달력 30일이 아니다. 금융 데이터 증강은 적용하지 않는다.

RNN/LSTM은 hidden=24, 1 layer, batch=64, AdamW lr=0.001, clip_norm=1, 최대 50 epoch, patience=8. 기본 모델은 표준화된 수준 예측, 잔차 LSTM은 마지막 입력에 변화량을 더하고 head를 0으로 초기화한다. 잔차 모델 weight_decay=0.01이다. Test 지표는 inverse transform 후 원 단위로 계산한다. MAPE는 |실제값|≤1e-8을 제외하고 제외 건수를 기록한다. 실제 환율 입력은 양수만 허용한다.

## 잔차 LSTM 추가 탐색 (종료)

기준 실험 이후 수행한 [하이퍼파라미터 탐색](../reports/residual-search/report.md)은 별도 실험이다. 3,600개 조합을 선별하고 상위 24개를 3개 시간 구간 × 3개 seed로 비교한 뒤, Validation으로 선택한 설정 하나를 Test에서 10개 seed로 평가했다. 각 학습 구간의 통계만 정규화에 사용하며, 체크포인트 선택 기준은 Validation MAE다. 학습 손실은 MAE/MSE를 비교했고 최대 60 epoch, patience 10을 적용했다.

Test 승리는 0회였으며 2회 동률·8회 패배했다. 평균 MAE는 Naive보다 약 0.72% 나빴다. 추가 탐색은 종료하며, 이 설정을 기존 실험의 기본값으로 채택하지 않는다. 이미 기준 실험에서 Test를 관찰한 이력이 있으므로 완전히 새로운 holdout 결과로 해석하지 않는다. 상세 탐색 범위·선정 규칙·불확실성·재현 명령은 위 보고서에 기록했다.

## 사람 검수

error_review.csv의 suggested_tag는 자동 가설이며 human_tag와 별개다. 사람의 오분류 사례 검토는 권장되는 선택적 오류 분석 절차다. 검토한 고유 이미지에 관찰한 원인의 human_tag를 작성할 수 있다. 태그는 dark_lighting, blur, low_resolution, background_clutter, occlusion, class_similarity, suspected_label_error, high_confidence_error, model_limitation 중 하나다. 과도한 해석 없이 불확실성을 기록한다. 검수 명령은 태그 형식만 확인하지만 실제로 사람이 봤는지 인증할 수는 없다.

수정 후 diagnostics review --csv <경로>로 HTML/PNG 갤러리와 집계를 갱신하고 report 명령으로 보고서에 반영할 수 있다. 허용된 human_tag가 작성된 사례만 검수 건수로 집계한다. 자동 suggested_tag만 있는 사례는 사람 검수가 아니다. 검수 통계는 분석 보조 정보이며 최소 건수나 완료 임계값은 없다. 검수가 0건이어도 정상 종료하며 중복 sample_id나 잘못된 human_tag 등 데이터 오류는 실패한다. 보고서는 두 실험 트랙의 audit.json과 필수 지표·학습 이력·Test 예측 산출물을 검증한 뒤 생성하며, 사람 검수 건수는 완료 조건이 아니다.

## 실패 대응 및 재현

- prepare만 온라인 접근한다. 다운로드 timeout 60초, 최대 3회, 1/2초 backoff. 실패 시 nonzero exit, 오류 로그 출력. 합성 데이터나 무작위 가중치로 자동 대체하지 않는다. CIFAR 아카이브는 torchvision checksum 검증을 사용한다.
- 시계열은 저장소에 포함한 공식 원본 표의 고정 CSV를 검증하여 임시 파일에 복사한 뒤 rename한다. 최초 FRED CSV 다운로드는 404, 원출처 직접 요청은 403을 반환해 격리된 웹 읽기로 공식 표를 확보했다. 데이터 출처와 수집 방식을 datasets/provenance.json에 기록했다. 입력 CSV를 직접 제공해도 같은 검증을 적용한다.
- 실험 출력 디렉터리는 이미 존재하면 중단한다. 원본·기존 결과·사람 태깅을 덮어쓰지 않는다. 실패 출력은 남겨 진단하고 새 경로에서 재시도한다. audit.json은 전체 track 성공 후에만 생성한다.
- 두 audit.json이 없는 불완전 결과로 종합 리포트를 생성하지 않는다. GPU가 없어도 CPU 실행이 가능하다. 기본 실험 명령은 자동 탐색을 수행하지 않으며, 별도 탐색 스크립트의 완료 결과는 위에 기록했다.
- seed, 입력 hash, split membership, 패키지 잠금 파일, epoch history를 보존한다. checkpoint는 기본적으로 저장하지 않으며(`save_checkpoints=False`), CLI에서 `--save-checkpoints`를 지정할 때만 `.pt` 파일로 저장한다. 다른 플랫폼·패키지·thread 설정 사이 bitwise 재현성은 보장하지 않는다.
- 기준 실험은 단일 seed의 교육용 비교이며 추가 탐색의 반복 seed 결과는 별도로 기록한다. 금융 투자 판단이나 통계적 유의성 주장을 하지 않는다. 실측 열세를 숨기지 않는다.

## 모듈 입출력

| 모듈 | 입력 → 출력 |
| --- | --- |
| prepare | 공개 데이터 → 검증된 로컬 데이터·출처 manifest |
| vision | CIFAR cache·seed·표본 수 → 모델·분할표·지표·실제 오분류 |
| timeseries | 단일 CSV·window → baseline/RNN/LSTM·지표·날짜별 예측 |
| review | 사람이 수정한 CSV → 검수 통계·태그별 HTML/PNG 갤러리 |
| report | 성공한 두 실험 디렉터리 → Markdown·비교 그래프·오류 갤러리 |
| common | seed·표·학습 이력 → 재현 설정·JSON·loss 그래프 |
