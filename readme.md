## Todo List

- [x] AVA 데이터셋 불러오는 포맷을 YOLO와 동일하게 바꾸는 함수 업데이트 (Update: 2023-08-01)
- [x] wandb 훈련 파라미터 모니터링 (Update: 2023-08-01)
- [x] 테스트 스크립트 제작 (Update: 2023-08-02)
- [x] 대조군 모델 훈련 및 테스트 가능 ; 따로 저장해둔 feature를 적용하지 않는 버전 (Update: 2023-08-02)
- [x] amp.Scaler (모델 훈련 속도 2배~) 사용불가능 (Update: 2023-08-03)
- [x] 모델 더 가볍게 만들기 (Update: 2023-08-03)
- [x] 대조군 모델 성능 파악 및 디버깅 진행중 (Update: 2023-08-03)
- [x] 훈련 및 테스트 결과 train_loss는 떨어지지만 val_loss는 0인 현상 발견 (기존 모델로는 해결 안 됨. Update: 2023-08-10)
- [x] YOWO (freeze) 후 Head를 붙여서 실험 후 성공 (Update: 2023-08-14)
- [x] loss 계산 시 target box 와 anchor의 매칭 가능성 (100%)로 확대 + cls loss만 적용 (Update: 2023-08-18)
- [x] 훈련 중 모델의 output 확인할 수 있는 모니터링 기능 보완 (Update: 2023-08-29)
- [X] YOWO에서는 NMS 시 class 가 달라도 IoU가 threshold 이상이면 없애는 조금은 다른 로직을 사용하는 것을 발견.. (정석으로 갈지, YOWO대로 갈지 고민중)
- [x] 코드 1차 정리 일단 TODO 리스트는 만들어놓았음 (Update: 2023-09-15)
- [] (1) test AVA, 예측과 타겟의 bbox 스케일 확인
- [x] (2) loss_ava에서 입력되는 인풋에 따라 build_target에서 anchor 고르는 방법 달리하기. (Update: 23-09-19, 코드에 에러없이 훈련됨, 하지만 진짜 훈련이 되는지는 의문)
- [] (3) 디버깅 하면서 loss 확인 및 하이퍼파라미터 튜닝

2023-09-19: 트레이닝 코드는 일단 돌아감. 근데 clo_loss가 잘 안 떨어짐. 
(1) Mask를 만드는 loss function에 문제가 있거나
(2) 거리를 좁히면서 훈련시키는 feature에 문제가 있거나
(3) 또는 훈련량이 부족해서 epoch를 늘리면 해결될수도..