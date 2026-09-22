"""Reports are rendered from measured CSVs, never from example scores."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import markdown_table, write_json
from .review import gallery


def run(root):
    root = Path(root)
    image = root / "vision"
    series = root / "timeseries"
    # Completion manifests written only after each track succeeds.
    ia = json.loads((image / "audit.json").read_text())
    ta = json.loads((series / "audit.json").read_text())
    im = pd.read_csv(image / "metrics.csv")
    tm = pd.read_csv(series / "metrics.csv")
    prediction = pd.read_csv(series / "predictions.csv", parse_dates=["date"])
    status = gallery(image / "error_review.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prediction.date, prediction.actual, label="Actual", color="tab:blue")
    for model, color in [
        ("LSTM", "tab:red"),
        ("LSTM_residual", "tab:green"),
        ("Naive", "gray"),
    ]:
        ax.plot(prediction.date, prediction[model], label=model, alpha=0.8, color=color)
    ax.set(xlabel="Date", ylabel="KRW per USD", title="Rolling one-step Test forecasts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(series / "predictions.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        ax.bar(tm.model, tm[metric])
        ax.set(ylabel=metric, title=f"Test {metric} (KRW per USD)")
        ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(series / "baseline_comparison.png", dpi=150)
    plt.close(fig)
    improvements = []
    for neural in ["RNN", "LSTM", "LSTM_residual"]:
        for baseline in [
            n for n in tm.model if n not in ["RNN", "LSTM", "LSTM_residual"]
        ]:
            rows = tm.set_index("model")
            improvements.append(
                {
                    "model": neural,
                    "baseline": baseline,
                    **{
                        m: 100
                        * (rows.loc[baseline, m] - rows.loc[neural, m])
                        / rows.loc[baseline, m]
                        for m in ["MAE", "RMSE", "MAPE"]
                    },
                }
            )
    comparison = pd.DataFrame(improvements)
    comparison.to_csv(series / "improvements_percent.csv", index=False)
    (root / "baseline_comparison.md").write_text(
        "# 베이스라인 비교\n\n동일 Test 날짜, 동일 원 단위. 개선률 = 100 × (베이스라인 오차 − 모델 오차) / 베이스라인 오차. 음수는 악화이며 우수성을 주장하지 않는다.\n\n"
        + markdown_table(tm)
        + "\n\n## 베이스라인별 개선률 (%)\n\n"
        + markdown_table(comparison)
        + "\n\n![Comparison](timeseries/baseline_comparison.png)\n"
    )
    diagnoses = []
    for folder, models in [
        (image, im.model.unique()),
        (series, ["RNN", "LSTM", "LSTM_residual"]),
    ]:
        for model in models:
            history = pd.read_csv(folder / f"{model}_history.csv")
            row = history.loc[history.Validation.idxmin()]
            ratio = row.Validation / max(row.Train, 1e-8)
            # Explicit relative heuristic, not a claim to identify irreducible error.
            diagnosis = (
                "High Variance 또는 분포 이동 의심"
                if ratio > 1.5
                else "큰 일반화 격차 없음; 절대 오차·베이스라인과 함께 편향 판단"
            )
            diagnoses.append(
                {
                    "model": model,
                    "best_epoch": int(row.epoch),
                    "Train_loss": row.Train,
                    "Validation_loss": row.Validation,
                    "gap": row.Validation - row.Train,
                    "interpretation": diagnosis,
                }
            )
    diagnosis = pd.DataFrame(diagnoses)
    diagnosis.to_csv(root / "loss_diagnosis.csv", index=False)
    selected = im[im.split == "Validation"].sort_values("loss").iloc[0].model
    test = im[im.split == "Test"].set_index("model")
    gain = (test.loc["augmented", "accuracy"] - test.loc["fine_tune", "accuracy"]) * 100
    transfer = (
        test.loc["fine_tune", "accuracy"] - test.loc["scratch", "accuracy"]
    ) * 100
    residual = tm.set_index("model").loc["LSTM_residual", "MAE"]
    lstm = tm.set_index("model").loc["LSTM", "MAE"]
    rnn = tm.set_index("model").loc["RNN", "MAE"]
    relative = 100 * (lstm - residual) / lstm
    drift = prediction.actual.mean() - ta["mean"]
    document = f"""# 성능 진단 리포트

## 실험 및 상태

Seed {ia["seed"]}, 이미지 클래스 {", ".join(ia["classes"])}. 클래스당 Train {ia["shots"]}장, Validation {ia["validation_per_class"]}장, Test {ia["test_per_class"]}장. 원본 이미지의 식별자와 SHA-256을 membership.csv에 기록하고 분할 간 동일 이미지 바이트 중복을 검사했다. 시계열은 DEXKOUS {ta["observations"]}개 관측이다. 두 트랙 모두 Validation loss로 checkpoint를 선택했다. 모든 비교군과 증강 설정은 실행 전에 고정했다.

사람 검수: **{status["reviewed"]}/{status["total_errors"]}건**. 사람 검수는 선택적인 오류 분석 절차이며 집계는 분석 보조 정보다. 실험 완료 여부는 두 트랙의 정상 완료로 판단한다. 자동 suggested_tag는 가설이며 실제 원인 또는 human review로 간주하지 않는다.

## 이미지 실측 비교

{markdown_table(im)}

Validation loss 기준 선택 전략: **{selected}**. 동일 예산의 scratch 대비 Fine-tuning Test 정확도 차이: **{transfer:.2f}%p**. 증강 + weight decay 적용은 기본 Fine-tuning 대비 **{gain:.2f}%p**다. 증강 실험은 사전 지정한 민감도 실험이며 사람의 원인 태깅으로 검증한 인과적 개선이 아니다. 한 seed·적은 epoch의 결과이므로 최적의 freezing 전략으로 일반화할 수 없다. 사전학습의 외부 ImageNet 데이터와 추가 학습 비용도 비교 한계다.

## 학습 곡선과 편향·분산

{markdown_table(diagnosis)}

곡선은 증강 없는 Train 평가와 Validation 평가를 동일한 eval 모드에서 측정했다. 이미지 loss는 cross entropy, 시계열 loss는 Train 통계로 표준화한 값의 MSE다. Validation/Train > 1.5는 격차를 찾기 위한 휴리스틱일 뿐 확정 진단이 아니다. 무작위 3-class 분류의 cross entropy 기준은 ln(3) ≈ 1.099다. Train·Validation이 모두 이 수준이고 정확도도 1/3 근처면 높은 편향이나 학습 부족을 의심한다. Train만 낮고 Validation이 높으면 소표본 과적합을 의심하고 증강·weight decay·early stopping을 비교한다. 분포 이동도 같은 격차를 만들 수 있다.

"""
    for model in im.model.unique():
        document += f"![{model} Train/Validation loss](vision/{model}_loss.png)\n\n"
    for model in ["RNN", "LSTM", "LSTM_residual"]:
        document += f"![{model} Train/Validation loss](timeseries/{model}_loss.png)\n\n"
    document += f"""## 시계열 실측과 개선

{markdown_table(tm)}

기본 LSTM 대비 마지막 관측값에 변화량을 더하는 LSTM_residual의 MAE 개선률은 **{relative:.2f}%**다. 잔차 모델은 Naive를 초기 예측으로 두고 변화량만 학습한다. 금융 시계열에 jittering/time-warping을 적용하지 않았다. Validation MAE 기준 베이스라인 선택은 **{ta["selected_baseline"]}**이며 Test로 파라미터를 고르지 않았다. [각 베이스라인 대비 개선률](baseline_comparison.md)의 음수도 그대로 보고한다. 딥러닝이 Naive보다 나쁘다면 복잡성 증가를 정당화할 수 없다.

![예측과 실제](timeseries/predictions.png)

![베이스라인 비교](timeseries/baseline_comparison.png)

## 데이터 특성 및 모델 구조

이미지는 클래스 균형을 맞췄으므로 이 실험의 클래스 불균형은 없다. 원본 32×32 영상을 128×128로 확대해도 새로운 세부 정보가 생기지 않는다. cat/deer/dog의 형태적 유사성과 배경 의존 가능성을 실제 오류 이미지와 클래스별 혼동 통계로 확인해야 한다. 낮은 해상도 자체가 모든 실패의 원인이라는 결론은 내리지 않는다.

시계열 Test 평균은 Train 평균보다 {drift:.2f} KRW/USD 차이가 난다. 표준화된 가격 수준을 직접 예측하는 신경망이 학습 범위 밖 수준으로 일반화하기 어려운지 확인할 근거다. Naive는 최근 수준을 즉시 반영하며 SMA는 변동을 완화하지만 급격한 변화에 늦게 반응한다. 지표는 환율 거래일 1-step rolling 평가이며 전체 Test를 한 번에 예측하는 장기 예측이 아니다.

CNN은 공간상의 국소 패턴을 공유 필터로 학습한다. RNN은 순서대로 hidden state를 갱신하므로 시계열에 적합하지만 긴 역전파에서 gradient 소실·폭주가 생길 수 있다. LSTM은 input/forget/output gate와 cell state를 추가해 장기 정보 유지 경로를 만든다. 이번 동일 30-step 실험의 RNN MAE는 {rnn:.4f}, LSTM MAE는 {lstm:.4f}다. 이 결과만으로 장기 기억의 우월성을 증명하지는 않는다. 장기 기억 효과를 분리하려면 더 긴 window 및 반복 seed 실험이 추가로 필요하다.

## 오류 진단 및 사람 검수

![실제 Validation 오분류 사례](vision/error_gallery.png)

[오분류 갤러리](vision/error_gallery.html)와 [분석표](vision/error_review.csv)는 기본 Fine-tuning의 실제 Validation 오분류 전체를 포함한다. suggested_tag는 밝기·신뢰도 기반 가설이다. 원한다면 실제 관찰에 따른 human_tag를 적고 diagnostics review로 갤러리와 통계를 갱신한다. 허용된 human_tag가 작성된 고유 사례만 사람 검수 및 태그별 건수에 포함한다. 검수 건수는 모델 실험 완료 조건이 아니다.

낮은 Train 정확도와 클래스 전반의 혼동은 모델/학습 예산 문제를 먼저 점검한다. 낮은 Train loss와 특정 배경·조명에 집중한 Validation 오류는 데이터 다양성 문제를 점검한다. 라벨 오류 의심은 원본 대조 후 별도 기록하고 Test 라벨을 수정해 성능을 높이지 않는다. 사람 태그의 상위 실패 원인을 바탕으로 다음 개선을 선택하고 새로운 실험 디렉터리에 기록해야 사람 분석 → 개선의 사이클이 완성된다.

## 재현 및 누수 근거

[누수 방지·실행 정책](../../docs/protocol.md), 각 트랙 audit.json, predictions.csv, history.csv, 이미지 membership.csv 및 [베이스라인 비교](baseline_comparison.md)를 함께 확인한다. 학습 가중치와 원본 데이터는 로컬 data/ 및 결과 디렉터리에 남기되 Git에는 포함하지 않는다.
"""
    (root / "diagnosis.md").write_text(document)
    conf = []
    for strategy in im.model.unique():
        table = pd.read_csv(image / f"{strategy}_Test_predictions.csv")
        for actual in range(3):
            for predicted in range(3):
                conf.append(
                    {
                        "model": strategy,
                        "actual": ia["classes"][actual],
                        "predicted": ia["classes"][predicted],
                        "count": int(
                            (
                                (table.actual == actual)
                                & (table.predicted == predicted)
                            ).sum()
                        ),
                    }
                )
    pd.DataFrame(conf).to_csv(image / "confusion_counts.csv", index=False)
    write_json(
        root / "status.json",
        {
            "experiments_complete": True,
            "human_review": status,
        },
    )
