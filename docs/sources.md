# 데이터·모델 출처

- Board of Governors of the Federal Reserve System, [H.10 Historical Rates for the South Korean Won](https://www.federalreserve.gov/releases/h10/hist/dat00_ko.htm): 2020-01-01~2024-12-31의 일별 KRW/USD, 휴일 ND 제외. 2026-09-21 격리 웹 읽기로 수집했다. 출처 표의 1,305개 평일 행을 누락 없이 읽어 1,249개의 유효 관측으로 저장했다. FRED DEXKOUS와 동일한 원출처 시계열이다.
- Torchvision, [CIFAR10 dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html): 공식 Train/Test 분할과 다운로드 checksum을 이용했다. 선택 클래스는 cat, deer, dog다. 원본 데이터는 저장소에 재배포하지 않고 실제 오분류 사례의 소형 이미지와 sample ID만 리포트에 포함한다.
- Torchvision, [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): IMAGENET1K_V1 사전학습 가중치. 공식 입력 변환은 256 resize/224 crop이며 이번 CPU 실험은 명시적으로 128 resize를 사용한다. mean/std는 공식 값을 유지한다.

외부 페이지의 제목·기관·용도를 2026-09-21 확인했다. CSV의 수집 경로와 SHA-256은 provenance.json에 기록했다. 임의 생성 시계열이나 임의 성능 수치는 사용하지 않았다.
