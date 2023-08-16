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

## Model Performance and Details

### Experiment 1. Freeze YOWO and learn the cloth information using only 'deepfashionv2' dataset

- [x] Model 01: YOWO with CFAM (Completed: 2023-08-15)

After freezing the YOWO model, features are extracted post the CFAM module. These features are then passed through the Class Head module, specifically tailored for classification tasks. In the diagram below, the term "Multi-head" implies the ability to handle multiple tasks, such as bounding box regression for a person and action classification.

![스크린샷 2023-08-16 204312](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/assets/96426723/74bf70df-08c3-4ed4-b9cd-173d380cc94b)

**Performance Metrics on deepfashionv2 validation set**:

| Metric   | Score  |
| -------- | ------ |
| Precision| 0.1968 |
| Recall   | 0.7282 |
| mAP0.5   | 0.1263 |


**Note**: The confidence threshold was set to 0.001 and the IOU threshold was set to 0.6. Given that deepfashionv2 boxes represent clothing and predicted bounding boxes encompass humans, achieving the IOU threshold is challenging, which results in the observed performance.

**Example of prediction**:
Here is the example of the prediction during test.
<p float="left">
  <img src="https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/assets/96426723/0309ea0b-2acc-49bb-a3a3-118c490d23a4" width="30%" />
  <img src="https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/assets/96426723/a701f894-929d-448a-91de-456f3a737bc2" width="30%" />
  <img src="https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/assets/96426723/9a5aab63-2ee1-4808-a309-9d80f914c343" width="30%" />
</p>

---

- [ ] Model 02: YOWO with 2D-backbone (In Progress)

For this iteration, we're extracting features after the 2D-backbone of YOWO. The features are then processed by the Class Head module to classify clothing in the deepfashionv2 dataset.

![스크린샷 2023-08-16 204258](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/assets/96426723/34b5b1dd-f2cc-42c0-9b20-724fcd282410)

**Performance Metrics**:

_The results will be updated upon completion._

---

**Example of prediction**:
Here is the example of the prediction during training.
<p float="left">
  <img src="https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/assets/96426723/8d0504b1-7cfe-4ac5-b7f7-05435f68df39" width="30%" />
  <img src="https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/assets/96426723/1fecc000-7eae-4a44-8589-5f97e6128ef4" width="30%" />
  <img src="https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/assets/96426723/ccd69beb-5ef1-4b68-b7a8-033099a59a2c" width="30%" />
</p>



## WanDB

To see the results, please follow the link below:

[WandB Project Results](https://wandb.ai/jaewon012754/YOLOR/runs/t7ax9mea?workspace=user-jjw012754)
