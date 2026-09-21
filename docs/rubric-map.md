# 평가 기준별 근거

| 항목 | 구현·산출물 | 상태/한계 |
| --- | --- | --- |
| 1 전이학습 전략 | vision.py, vision/metrics.csv | 동일 split·초기 seed·예산, 학습률 차이 명시 |
| 2 베이스라인 | timeseries.py, baseline_comparison.md | 7종 베이스라인 및 모든 개선률 |
| 3 편향·분산 | diagnosis.md, loss_diagnosis.csv, 각 loss PNG | 비율 휴리스틱과 절대 수준 함께 설명 |
| 4 오분류 | error_review.csv, error_gallery.html | 실제 사례 확보, 사람 태깅 대기 |
| 5 시각화 | loss PNG, predictions.png, baseline_comparison.png, gallery | 필수 4종 생성 |
| 6 문맥 혼합 | 단일 ticker 검사, 독립 hidden state, tests/test_contracts.py | 다른 샘플 변경 시 예측 불변 |
| 7 시간순 분할 | prepare_series, timeseries/audit.json | 원시 관측 7/1/2 |
| 8 이미지 전처리 | preprocessing, docs/protocol.md | 크기·평균·표준편차·증강 명시 |
| 9 모듈화 | src/diagnostics, docs/protocol.md | 입출력 표 |
| 10 정규화 누수 | Train prefix mean/std, 계약 테스트 | Test 변경에도 통계 불변 |
| 11 High Variance | 증강·weight decay·early stopping 비교 | 사람 오류 원인에 따른 추가 개선은 대기 |
| 12 Few-shot 효용 | scratch/linear_probe/fine_tune 실측 | 40장/class, 사전학습 외부 데이터 한계 |
| 13 Look-ahead | shift(1), [t-window,t), 미래 변조 테스트 | 모든 baseline 입력은 target 이전 |
| 14 치명적 실패 | prepare.retry, CLI nonzero exit, 새 출력 경로 | timeout/retry/캐시·CSV 사용 정책 |
| 15 데이터 특성 | diagnosis.md, confusion_counts.csv | 균형 표본·저해상도·시계열 수준 이동 |
| 16 RNN/LSTM | 동일 window·hidden 실험과 구조 설명 | 장기 기억 자체의 우월성은 미입증 |
| 17 누수 구간 | docs/protocol.md 누수 표 | 분할·scaler·window·feature·모델 선택 |
| 18 원인 구분 | diagnosis.md 오류 진단 절차 | 사람 직접 관찰 완료 후 원인별 통계 확정 |

최종 제출 미완료 항목: 최소 30건 사람 검수 및 근거 기반 개선 사이클, GitHub 원격 게시. 코드·수치·자동 태그만으로 해당 항목을 완료 처리하지 않는다.
