# human_tag 수정 초안 — 사람 재검수 대기

사용자 요청에 따라 AI가 기존 장별 시각 검수와 사용자 기준을 바탕으로 94건을 다시 분류했다. **76건을 변경했고 18건은 기존 태그를 유지했다.** 현재 human_tag는 사람이 추후 확정할 초안이다.

원본은 모두 32×32다. 앞선 검수에서 94장 모두를 ID별로 확대 확인했고, 이번 수정은 그때의 장별 관찰에 근거한다. 확대가 세부를 복원하는 것은 아니다. 실제 오분류 원인을 입증하는 분석은 아니다.

## 분류 기준

- low_resolution: 사람도 식별에 필요한 세부를 읽기 어려움. 모든 오분류를 해상도 문제로 묶지는 않음.
- model_limitation: 사람이 구별할 형태 단서가 남아 있지만 모델이 틀림. 특정 특징을 학습하지 못했다는 뜻으로 단정하지 않음.
- class_similarity: 귀·얼굴·자세 등 닮아 보이는 단서가 있음. 원인으로 확정하지 않음.
- background_clutter / occlusion: 배경과 경계가 섞이거나 물체에 가려지는 현상이 관찰됨.
- high_confidence_error는 점수 설명이라 이번 원인 중심 초안에서는 대체함. 예측 confidence 값은 보존함.
- dark_lighting은 검은 배경·털색과 조명 부족을 구분할 증거가 약해 다른 관찰에 맞게 대체함.

19434는 low_resolution, 162와 12253은 model_limitation으로 반영했다.

## 초안 분포

| 태그 | 건수 |
| --- | ---: |
| background_clutter | 9 |
| class_similarity | 24 |
| low_resolution | 33 |
| model_limitation | 27 |
| occlusion | 1 |

## 재검수 방법

[이미지·변경 전후 태그·근거 보기](tag_audit.html)에서 확인한 뒤 [error_review.csv](error_review.csv)의 human_tag와 review_note를 수정하고, 확정한 행의 tag_review_status를 human_confirmed로 바꾸면 된다. review_note는 현재 AI가 작성한 메모다.

**기존 review/report 명령은 tag_review_status를 읽지 않고 채워진 human_tag를 모두 검수 건수로 집계한다.** 아직 사람의 재검수가 끝나지 않았으므로 이번에는 공식 리포트와 오류 갤러리를 재생성하지 않았다. 기존 리포트의 태그 통계는 이전 확정본 기준이다.

검수 CSV의 previous_human_tag, assessment, visual_observation, assessment_reason은 수정 전 검수 기록이며, human_tag와 review_note는 현재 초안이다.

## 장별 변경 및 근거

| 이미지 | 변경 전 | 현재 초안 | 관찰 및 분류 근거 |
| --- | --- | --- | --- |
| [cifar10_train_4025](errors/cifar10_train_4025.png) | class_similarity | class_similarity | 밝은 둥근 얼굴과 갈색 몸통이 보이고 귀·주둥이 세부가 뭉개져 있다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_47823](errors/cifar10_train_47823.png) | high_confidence_error | background_clutter | 갈색 동물이 실내 물건 사이에 있고 얼굴과 배경의 경계가 흐리다. 배경의 물건·색·무늬와 피사체 경계가 섞이는 점을 우선 기록. 모델이 배경을 사용했는지는 미확인. |
| [cifar10_train_43254](errors/cifar10_train_43254.png) | high_confidence_error | low_resolution | 몸을 낮춘 어두운 동물의 옆모습이며 얼굴이 작고 몸통 무늬와 섞인다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_14485](errors/cifar10_train_14485.png) | class_similarity | background_clutter | 흰색과 검은색 피사체 주변에 여러 색의 물건이 겹쳐 윤곽이 불명확하다. 배경의 물건·색·무늬와 피사체 경계가 섞이는 점을 우선 기록. 모델이 배경을 사용했는지는 미확인. |
| [cifar10_train_48443](errors/cifar10_train_48443.png) | dark_lighting | low_resolution | 갈색 피사체가 녹색 풀 위에 서 있고 몸통이 배경보다 어둡다. 어두운 몸과 작은 얼굴에서 구별 세부가 부족함. 털색과 배경 대비의 영향을 조명 부족으로 단정하지 않음. |
| [cifar10_train_43339](errors/cifar10_train_43339.png) | high_confidence_error | low_resolution | 밝은 바닥 위 작은 전신이며 얼굴을 구성하는 픽셀이 적다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_11386](errors/cifar10_train_11386.png) | high_confidence_error | model_limitation | 어두운 배경 앞 얼굴과 흰 가슴이 보이며 뾰족한 귀 윤곽이 남아 있다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_32172](errors/cifar10_train_32172.png) | class_similarity | class_similarity | 갈색·흰색 몸이 둥글게 말려 있고 얼굴과 몸통의 경계가 불분명하다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_47667](errors/cifar10_train_47667.png) | class_similarity | model_limitation | 파란 배경 앞 전신과 위로 든 긴 꼬리가 보이고 얼굴은 작다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_17024](errors/cifar10_train_17024.png) | high_confidence_error | background_clutter | 갈색 피사체가 비슷한 색의 바닥·가구 앞에 있어 몸과 배경의 대비가 낮다. 배경의 물건·색·무늬와 피사체 경계가 섞이는 점을 우선 기록. 모델이 배경을 사용했는지는 미확인. |
| [cifar10_train_38229](errors/cifar10_train_38229.png) | class_similarity | low_resolution | 회색 바닥의 흰색·검은색 작은 덩어리 형태로 얼굴 방향도 불명확하다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_39313](errors/cifar10_train_39313.png) | high_confidence_error | model_limitation | 줄무늬 몸과 얼굴이 보이고 오른쪽 구조물 및 배경과 가깝게 맞닿아 있다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_25077](errors/cifar10_train_25077.png) | class_similarity | model_limitation | 뾰족한 귀와 줄무늬 얼굴이 크게 보이며 일부 얼굴은 그림자에 있다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_2505](errors/cifar10_train_2505.png) | class_similarity | class_similarity | 붉은 바닥에 누운 동물의 옆얼굴과 몸이 작고 흐리게 보인다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_1041](errors/cifar10_train_1041.png) | class_similarity | background_clutter | 밝은 몸통이 주변의 붉고 어두운 물건들과 겹치며 얼굴 위치가 불명확하다. 배경의 물건·색·무늬와 피사체 경계가 섞이는 점을 우선 기록. 모델이 배경을 사용했는지는 미확인. |
| [cifar10_train_41540](errors/cifar10_train_41540.png) | class_similarity | model_limitation | 앉은 흰 가슴의 동물과 얼굴 윤곽이 보이며 주변에 수직 구조물이 있다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_5473](errors/cifar10_train_5473.png) | high_confidence_error | model_limitation | 풀 위를 걷는 전신이며 긴 다리 윤곽에 비해 머리는 작다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_19780](errors/cifar10_train_19780.png) | class_similarity | model_limitation | 단색 배경 앞 앉은 동물의 뾰족한 귀와 긴 꼬리가 보인다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_34011](errors/cifar10_train_34011.png) | high_confidence_error | low_resolution | 길과 수풀 사이 작은 전신이며 꼬리는 올라가 있고 얼굴은 매우 작다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_28574](errors/cifar10_train_28574.png) | class_similarity | occlusion | 검은 동물이 밝은 상자 뒤에 있고 귀는 보이나 하체는 가려져 있다. 상자 등 주변 물체에 피사체 일부가 가려져 전신 단서가 부족한 점을 기록. |
| [cifar10_train_15876](errors/cifar10_train_15876.png) | high_confidence_error | low_resolution | 밝은 바닥 위 어두운 웅크린 몸이며 얼굴 방향과 세부가 잘 보이지 않는다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_45752](errors/cifar10_train_45752.png) | high_confidence_error | low_resolution | 밝은 배경에 흰색·검은색 얼굴 또는 상체가 보이며 경계가 뭉개져 있다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_1271](errors/cifar10_train_1271.png) | class_similarity | low_resolution | 갈색 동물이 앉아 있고 얼굴과 다리가 흐리게 뭉개져 있다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_24652](errors/cifar10_train_24652.png) | high_confidence_error | class_similarity | 흰 털의 둥근 얼굴과 작은 눈·코가 정면으로 크게 보인다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_45209](errors/cifar10_train_45209.png) | class_similarity | class_similarity | 흰 얼굴과 어두운 귀·눈 주변이 보이며 주둥이 길이가 불명확하다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_4437](errors/cifar10_train_4437.png) | high_confidence_error | model_limitation | 녹색 풀 앞 주황색 전신이 옆으로 보이며 얼굴 세부가 적다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_9394](errors/cifar10_train_9394.png) | high_confidence_error | low_resolution | 수풀 앞 작은 갈색 피사체로 배경이 차지하는 면적이 크다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_46812](errors/cifar10_train_46812.png) | high_confidence_error | low_resolution | 밝은 줄무늬 몸 또는 얼굴이 화면을 채우지만 경계가 강하게 뭉개져 있다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_24031](errors/cifar10_train_24031.png) | class_similarity | class_similarity | 붉은 바닥에 엎드린 줄무늬 얼굴이 정면으로 보이며 귀가 뚜렷하지 않다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_28162](errors/cifar10_train_28162.png) | high_confidence_error | background_clutter | 줄무늬 배경과 가는 구조물 앞 갈색 동물이 앉아 있어 윤곽이 배경과 섞인다. 배경의 물건·색·무늬와 피사체 경계가 섞이는 점을 우선 기록. 모델이 배경을 사용했는지는 미확인. |
| [cifar10_train_34579](errors/cifar10_train_34579.png) | dark_lighting | low_resolution | 검은 배경 앞 어두운 몸통의 세부는 약하고 얼굴·가슴 일부와 아래 주황색 면은 밝다. 어두운 털·배경 사이에서 몸과 얼굴 세부가 부족함. 조명 자체가 원인인지는 불확실함. |
| [cifar10_train_24036](errors/cifar10_train_24036.png) | high_confidence_error | low_resolution | 갈색 바닥의 회색·흰색 피사체가 작은 덩어리로 보여 자세와 얼굴이 불명확하다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_36269](errors/cifar10_train_36269.png) | class_similarity | class_similarity | 주황색 몸이 말린 자세이며 옆얼굴과 귀 일부가 몸통에 겹친다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_41918](errors/cifar10_train_41918.png) | class_similarity | low_resolution | 회청색 무늬가 화면을 채우고 얼굴·몸의 경계를 안정적으로 읽기 어렵다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_28786](errors/cifar10_train_28786.png) | dark_lighting | model_limitation | 어두운 배경과 얼굴 상부에 비해 흰 가슴은 밝고 한쪽 눈 부근만 두드러진다. 귀와 얼굴·가슴 윤곽의 고양이 단서가 남아 있으나 dog으로 오분류. 밝은 가슴과 어두운 얼굴만으로 조명 원인을 확정하지 않음. |
| [cifar10_train_517](errors/cifar10_train_517.png) | high_confidence_error | model_limitation | 어두운 배경 앞 머리와 밝은 가슴이 보이고 몸 일부는 주변 구조물과 겹친다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_966](errors/cifar10_train_966.png) | high_confidence_error | class_similarity | 풀 위 정면 얼굴과 귀가 보이고 코 주변의 밝은 무늬가 두드러진다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_10489](errors/cifar10_train_10489.png) | class_similarity | class_similarity | 어두운 몸과 위로 뻗은 귀·머리 윤곽이 작은 크기로 보인다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_16155](errors/cifar10_train_16155.png) | high_confidence_error | model_limitation | 밝은 배경에 큰 귀와 가는 얼굴이 보이며 밝은 영역의 세부가 약하다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_27572](errors/cifar10_train_27572.png) | high_confidence_error | class_similarity | 정면 얼굴의 긴 귀와 눈·주둥이가 보이나 얼굴 내부가 픽셀 단위로 뭉개져 있다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_180](errors/cifar10_train_180.png) | high_confidence_error | low_resolution | 녹색 배경과 몸통의 색이 비슷하고 머리·목의 대비가 낮다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_26854](errors/cifar10_train_26854.png) | class_similarity | background_clutter | 갈색 배경 속 작은 정면 피사체로 얼굴과 몸의 경계가 희미하다. 배경의 물건·색·무늬와 피사체 경계가 섞이는 점을 우선 기록. 모델이 배경을 사용했는지는 미확인. |
| [cifar10_train_22981](errors/cifar10_train_22981.png) | high_confidence_error | model_limitation | 정면 머리와 양쪽으로 뻗은 귀·뿔 형태가 보이며 얼굴의 세부 대비가 낮다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_29632](errors/cifar10_train_29632.png) | class_similarity | low_resolution | 넓고 밝은 배경 아래에 피사체가 매우 작게 위치한다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_33245](errors/cifar10_train_33245.png) | class_similarity | model_limitation | 풀 위 옆모습의 긴 다리와 몸통이 보이나 머리는 작다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_41237](errors/cifar10_train_41237.png) | class_similarity | low_resolution | 푸른 배경 아래 어두운 실루엣만 작게 보인다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_21033](errors/cifar10_train_21033.png) | high_confidence_error | class_similarity | 갈색 정면 얼굴이 화면 대부분을 차지하고 코 주변이 검게 보인다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_28470](errors/cifar10_train_28470.png) | dark_lighting | model_limitation | 배경은 짙지만 밝은 갈색 몸통과 다리 윤곽은 구분된다. 몸통과 다리의 사슴 형태가 남아 있지만 cat으로 오분류. 검은 배경을 피사체 조명 부족으로 해석하지 않음. |
| [cifar10_train_15415](errors/cifar10_train_15415.png) | class_similarity | class_similarity | 옆얼굴과 길게 나온 주둥이가 보이고 귀·뿔 세부는 뭉개져 있다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_33215](errors/cifar10_train_33215.png) | class_similarity | model_limitation | 밝은 배경에 옆모습의 긴 목·다리·몸통이 보인다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_9775](errors/cifar10_train_9775.png) | class_similarity | model_limitation | 긴 목과 가는 다리의 옆모습이 보이고 얼굴은 작다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_24236](errors/cifar10_train_24236.png) | class_similarity | low_resolution | 갈색 몸통이 화면을 크게 차지하고 머리·다리는 부분적으로만 식별된다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_162](errors/cifar10_train_162.png) | dark_lighting | model_limitation | 검은 배경 앞 밝은 몸통과 다리가 보이고 고개를 숙이고 있다. 사용자 관찰대로 긴 목과 전신 형태에서 사슴 단서가 보이지만 cat으로 오분류. 목 특징을 학습하지 못했는지는 이 결과만으로 확인 불가. |
| [cifar10_train_33703](errors/cifar10_train_33703.png) | class_similarity | low_resolution | 몸통과 다리가 보이지만 머리 부분은 어둡고 배경 구조물과 섞인다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_14186](errors/cifar10_train_14186.png) | high_confidence_error | model_limitation | 회색 배경에 어두운 전신이 보이고 긴 목·귀 윤곽에 비해 얼굴 세부가 없다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_42313](errors/cifar10_train_42313.png) | class_similarity | class_similarity | 접힌 다리의 몸통과 작은 머리가 옆으로 보인다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_34425](errors/cifar10_train_34425.png) | class_similarity | low_resolution | 푸른 배경 속 작은 옆모습으로 머리와 다리가 몇 픽셀로 표현된다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_33037](errors/cifar10_train_33037.png) | high_confidence_error | low_resolution | 갈색 몸 또는 얼굴이 화면을 채우며 경계와 눈·귀 세부가 불명확하다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_12253](errors/cifar10_train_12253.png) | dark_lighting | model_limitation | 검은 배경 앞 밝은 목·몸통과 녹색 바닥이 분리되어 보인다. 사용자 관찰대로 목과 몸통에 사슴 형태 단서가 보이지만 dog으로 오분류. 배경이 검다는 이유로 저조도로 분류하지 않음. |
| [cifar10_train_10478](errors/cifar10_train_10478.png) | high_confidence_error | background_clutter | 가로로 뻗은 귀와 얼굴이 보이나 비슷한 색의 배경 무늬와 섞인다. 배경의 물건·색·무늬와 피사체 경계가 섞이는 점을 우선 기록. 모델이 배경을 사용했는지는 미확인. |
| [cifar10_train_47408](errors/cifar10_train_47408.png) | high_confidence_error | model_limitation | 큰 귀와 가는 얼굴이 정면으로 보이고 배경의 세로 구조물과 겹친다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_14555](errors/cifar10_train_14555.png) | high_confidence_error | low_resolution | 풀밭 위 고개를 숙인 전신이며 얼굴은 몸의 그늘 속에 있다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_30719](errors/cifar10_train_30719.png) | high_confidence_error | model_limitation | 갈색 정면 얼굴과 밝은 목 부분이 크게 보이고 귀 일부는 화면 가장자리에 걸린다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_45408](errors/cifar10_train_45408.png) | class_similarity | class_similarity | 얼굴과 목의 가까운 구도이며 긴 귀와 주둥이가 보인다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_19007](errors/cifar10_train_19007.png) | high_confidence_error | low_resolution | 밝고 둥근 털 덩어리와 어두운 아래 윤곽이 보이며 얼굴 세부가 약하다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_14034](errors/cifar10_train_14034.png) | class_similarity | low_resolution | 검은색·흰색 이미지에서 아래쪽 작은 동물과 주변 직선 구조물이 보인다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_19434](errors/cifar10_train_19434.png) | dark_lighting | low_resolution | 짙은 녹색 배경 앞 밝은 흰색 얼굴과 검은 코가 보인다. 사용자도 피사체 식별이 어렵다고 확인. 식별에 필요한 세부 부족을 기록하며 어두운 배경만으로 저조도라고 판단하지 않음. |
| [cifar10_train_26753](errors/cifar10_train_26753.png) | high_confidence_error | background_clutter | 청록색 주변 물체 사이에 밝은 얼굴이 있고 귀·몸의 경계가 불분명하다. 배경의 물건·색·무늬와 피사체 경계가 섞이는 점을 우선 기록. 모델이 배경을 사용했는지는 미확인. |
| [cifar10_train_6154](errors/cifar10_train_6154.png) | class_similarity | class_similarity | 흰색·검은색 무늬의 가는 전신과 올라온 머리가 보인다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_21920](errors/cifar10_train_21920.png) | high_confidence_error | low_resolution | 밝은 배경에서 옆으로 누운 갈색·흰색 몸과 머리가 보인다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_40772](errors/cifar10_train_40772.png) | class_similarity | class_similarity | 갈색 피사체가 옆으로 서 있고 다리는 가늘며 얼굴은 작다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_762](errors/cifar10_train_762.png) | class_similarity | class_similarity | 작은 얼굴에 매우 큰 직립 귀가 두드러진다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_12884](errors/cifar10_train_12884.png) | class_similarity | class_similarity | 흰 배경에 밝은 작은 얼굴과 옆으로 뻗은 귀가 보인다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_49467](errors/cifar10_train_49467.png) | high_confidence_error | class_similarity | 검은 얼굴과 큰 직립 귀가 정면으로 보이고 얼굴 내부 대비는 낮다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_21261](errors/cifar10_train_21261.png) | high_confidence_error | low_resolution | 풀밭의 어두운 몸과 얼굴이 하나의 덩어리처럼 보이며 코·귀 세부가 약하다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_33424](errors/cifar10_train_33424.png) | class_similarity | class_similarity | 갈색 얼굴과 몸이 비슷한 색의 배경에 붙어 있고 짧은 주둥이가 보인다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_10897](errors/cifar10_train_10897.png) | high_confidence_error | background_clutter | 푸른 무늬 배경에 밝은 얼굴이 있고 눈·코와 주변 무늬가 섞인다. 배경의 물건·색·무늬와 피사체 경계가 섞이는 점을 우선 기록. 모델이 배경을 사용했는지는 미확인. |
| [cifar10_train_26998](errors/cifar10_train_26998.png) | high_confidence_error | model_limitation | 풀 위 어두운 전신의 가는 다리와 옆으로 향한 머리가 보인다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_17152](errors/cifar10_train_17152.png) | high_confidence_error | low_resolution | 풀밭의 흰 전신을 위에서 내려다보는 구도로 얼굴은 작고 아래쪽에 있다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_37341](errors/cifar10_train_37341.png) | high_confidence_error | model_limitation | 세로로 긴 구도에 어두운 상체와 밝은 가슴이 보이고 얼굴은 작다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_5043](errors/cifar10_train_5043.png) | high_confidence_error | low_resolution | 밝은 얼굴과 주변 밝은 무늬가 섞여 눈·코 윤곽을 읽기 어렵다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_24585](errors/cifar10_train_24585.png) | high_confidence_error | class_similarity | 청록색 배경 앞 갈색 얼굴과 큰 귀가 선명한 외곽을 만든다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_19293](errors/cifar10_train_19293.png) | high_confidence_error | low_resolution | 어두운 얼굴과 큰 귀가 보이며 아래 몸통은 주변 물건과 겹친다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_7454](errors/cifar10_train_7454.png) | class_similarity | class_similarity | 갈색 털이 얼굴 주변을 채우고 짧고 검은 주둥이가 크게 보인다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_8452](errors/cifar10_train_8452.png) | class_similarity | model_limitation | 검은 정면 얼굴과 옆으로 내려간 귀, 흰 가슴이 보인다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_8836](errors/cifar10_train_8836.png) | class_similarity | low_resolution | 길고 밝은 털이 얼굴과 몸통 경계를 덮고 눈·코가 작게 보인다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_23523](errors/cifar10_train_23523.png) | high_confidence_error | low_resolution | 갈색 옆모습의 몸통이 크게 보이고 머리는 아래쪽으로 향해 얼굴이 작다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_40154](errors/cifar10_train_40154.png) | high_confidence_error | model_limitation | 강한 녹색 배경 앞 검은색·갈색 피사체의 정면 몸과 얼굴이 보인다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_38074](errors/cifar10_train_38074.png) | class_similarity | class_similarity | 밝은 얼굴과 뾰족한 귀가 정면으로 보이며 아래에 파란 물건이 있다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_7677](errors/cifar10_train_7677.png) | high_confidence_error | model_limitation | 흰 전신과 털로 둘러싸인 얼굴이 보이고 얼굴 내부 세부가 작다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
| [cifar10_train_21534](errors/cifar10_train_21534.png) | high_confidence_error | low_resolution | 흰 배경 앞 긴 흰 털의 앉은 몸이 보이며 얼굴을 구별하는 점들이 적다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_29127](errors/cifar10_train_29127.png) | class_similarity | class_similarity | 갈색 전신의 옆모습과 가는 다리가 보이고 머리는 프레임 오른쪽에 있다. 귀·얼굴 또는 전신 자세에서 다른 클래스와 비슷하게 읽힐 여지가 있어 유사성 가설로 분류. 실제 모델 판단 근거는 미확인. |
| [cifar10_train_33800](errors/cifar10_train_33800.png) | high_confidence_error | low_resolution | 풀밭에서 달리는 듯한 흰색·갈색 전신이며 옆얼굴이 매우 작다. 32×32 이미지에서 얼굴·몸의 구별에 필요한 세부를 충분히 읽기 어려워 분류. 촬영 흔들림이나 조명 부족을 별도로 단정하지 않음. |
| [cifar10_train_33768](errors/cifar10_train_33768.png) | high_confidence_error | model_limitation | 검은 얼굴이 화면을 채우고 눈·코의 대비가 낮지만 넓은 주둥이 윤곽이 남아 있다. 사람이 클래스를 구별할 형태 단서가 남아 있는데 모델이 오분류한 사례로 분류. 원인은 미확인이고 특정 특징을 학습하지 못했다고 단정하지 않음. |
