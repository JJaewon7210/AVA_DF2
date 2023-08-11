
<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>Combined_Learner_AVA2.2_and_Deepfashion2
</h1>
<h3>◦ Unlock Limitless Learning with AVA2.2 & Deepfashion2</h3>
<h3>◦ Developed with the software and tools listed below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Markdown-000000.svg?style&logo=Markdown&logoColor=white" alt="Markdown" />
</p>
<img src="https://img.shields.io/github/languages/top/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2?style&color=5D6D7E" alt="GitHub top language" />
<img src="https://img.shields.io/github/languages/code-size/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2?style&color=5D6D7E" alt="GitHub code size in bytes" />
<img src="https://img.shields.io/github/commit-activity/m/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2?style&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/license/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2?style&color=5D6D7E" alt="GitHub license" />
</div>

---

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [⚙️ Features](#-features)
- [📂 Project Structure](#project-structure)
- [🧩 Modules](#modules)
- [🚀 Getting Started](#-getting-started)
- [🗺 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

---


## 📍 Overview

The project aims to combine two datasets, AVA2.2 and DeepFashion2, for video analysis tasks such as action recognition or detection. It provides functionalities for loading and preprocessing images, sampling frames, and performing data augmentation techniques. Additionally, the project includes utility functions for evaluating AVA datasets, manipulating image paths and labels, and working with label maps in TensorFlow. The core value proposition of the project lies in its ability to efficiently preprocess and combine data from different datasets, enabling researchers and developers to perform more accurate and comprehensive video analysis tasks.

---

## ⚙️ Features

HTTPStatus Exception: 429

---


## 📂 Project Structure




---

## 🧩 Modules

<details closed><summary>Root</summary>

| File                                                                                                                                                                       | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---                                                                                                                                                                        | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [test_ava.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/test_ava.py)                                                               | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [test_df2.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/test_df2.py)                                                               | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [train.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/train.py)                                                                     | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [ava_dataset.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_dataset.py)                                                | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [ava_dataset_utils.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_dataset_utils.py)                                    | This code snippet comprises several functionalities. It includes functions for loading and preprocessing images for video analysis, sampling frames from videos, manipulating image paths and labels, and performing data augmentation techniques such as resizing, cropping, flipping, and color distortions. These functions are essential for various video analysis tasks such as action recognition or detection.                                                                                                                                                                                                    |
| [ava_eval_helper.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_eval_helper.py)                                        | This code snippet provides helper functions for evaluating AVA (Atomic Visual Action) datasets. It includes functions to load boxes and class labels from CSV files, read exclusions, read label maps, and run AVA evaluation given annotation/prediction files or numpy arrays. The evaluation includes calculating metrics such as precision and mean average precision (mAP) at an Intersection over Union (IOU) of 0.5. The code also includes functions to convert data formats and write prediction results in the official AVA format.                                                                             |
| [ava_helper.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_helper.py)                                                  | This code snippet provides functions for loading image paths, boxes, and labels from files in a specific format for a computer vision dataset. It also includes functions for extracting keyframe data and calculating box statistics.                                                                                                                                                                                                                                                                                                                                                                                    |
| [combined_dataset.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\combined_dataset.py)                                      | The code snippet defines a class called CombinedDataset, which serves as a custom dataset for PyTorch. It takes in two other datasets as inputs, dataset1 and dataset2. The dataset combines data from these two datasets and provides the functionality to access and preprocess the combined data efficiently. The collate_fn method is provided to preprocess and organize a batch of data for a deep learning model. The class also includes logging functionalities to print important information about the datasets being combined.                                                                                |
| [cv2_transform.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\cv2_transform.py)                                            | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [image.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\image.py)                                                            | The given code snippet contains various utility functions for image manipulation and transformation. It includes functions for flipping images, transforming coordinates, getting affine transformations, cropping images, calculating Gaussian radius, drawing Gaussian and MSRA heatmaps, and performing color augmentation. These functions can be used to preprocess and modify images for various computer vision tasks.                                                                                                                                                                                             |
| [transform.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\transform.py)                                                    | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [yolo_datasets.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\yolo_datasets.py)                                            | Prompt exceeds max token limit: 6408.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [label_map_util.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\label_map_util.py)                           | The provided code snippet is a collection of utility functions for working with label maps in TensorFlow. It includes functions for validating a label map, creating a category index from a label map, converting a label map to a list of categories, loading a label map from a file, and creating a label map dictionary.                                                                                                                                                                                                                                                                                             |
| [metrics.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\metrics.py)                                         | The provided code snippet includes functions for computing metrics like precision, recall, average precision, and CorLoc. It validates the input and computes the metrics using numpy arrays. It also handles edge cases and returns the results.                                                                                                                                                                                                                                                                                                                                                                         |
| [np_box_list.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\np_box_list.py)                                 | The provided code snippet defines a class called `BoxList` that represents a collection of bounding boxes. It provides functionalities to initialize the collection, add related fields, retrieve box coordinates, and check the validity of the boxes. The class is implemented using numpy arrays for efficient computation.                                                                                                                                                                                                                                                                                            |
| [np_box_list_ops.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\np_box_list_ops.py)                         | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [np_box_mask_list.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\np_box_mask_list.py)                       | This code snippet provides a BoxMaskList class that extends the np_box_list.BoxList class. It is a wrapper for boxes and masks, where masks correspond to the full image. The class constructor takes in numpy arrays for box coordinates and mask data. The class also provides a method to access the masks.                                                                                                                                                                                                                                                                                                            |
| [np_box_mask_list_ops.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\np_box_mask_list_ops.py)               | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [np_box_ops.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\np_box_ops.py)                                   | The provided code snippet contains operations for manipulating and analyzing bounding boxes represented as [N, 4] NumPy arrays. The core functionalities include computing areas of boxes, pairwise intersection-over-union (IOU) scores, and pairwise intersection-over-area (IOA) scores. The code employs numerical calculations to perform these operations efficiently.                                                                                                                                                                                                                                              |
| [np_mask_ops.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\np_mask_ops.py)                                 | This code snippet provides operations for manipulating [N, height, width] numpy arrays representing masks. It includes functionality for computing mask areas, pairwise intersection-over-union scores, and pairwise intersection-over-area scores. The code checks the dtype of the input arrays, raises a ValueError if the dtype is not np.uint8, and uses numpy operations to perform the desired computations.                                                                                                                                                                                                       |
| [object_detection_evaluation.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\object_detection_evaluation.py) | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [per_image_evaluation.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\per_image_evaluation.py)               | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [standard_fields.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/datasets\ava_evaluation\standard_fields.py)                         | The code snippet defines classes that specify naming conventions used for object detection. These classes define the names for various input, output, and data fields used in the object detection pipeline, such as image, bounding boxes, classes, scores, and more. These naming conventions help in standardizing the data representation and communication across different parts of the pipeline.                                                                                                                                                                                                                   |
| [BiFPN.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/model\BiFPN.py)                                                               | The provided code snippet is a implementation of a Bi-directional Feature Pyramid Network (BiFPN) using PyTorch. It consists of different modules such as DepthwiseConvBlock, ConvBlock, and BiFPNBlock. The BiFPNBlock performs depthwise separable convolution and combines multiple feature maps at different scales. The BiFPN module takes input feature maps at different levels and applies the BiFPNBlock to generate enhanced feature maps. The purpose of this network is to improve the performance of object detection tasks by integrating features from multiple scales.                                    |
| [cfam.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/model\cfam.py)                                                                 | The provided code snippet implements a Channel Attention Module (CAM_Module) and a modified version of it called CFAMBlock. The CFAMBlock consists of convolutional layers with batch normalization and ReLU activation, followed by the CAM_Module. The CFAMBlock takes an input tensor, applies the convolutional layers, performs channel attention using the CAM_Module, and outputs a tensor. The code also includes a main function that creates a sample input tensor and tests the CFAMBlock by passing the input through it and printing the output tensor size.                                                 |
| [darknet.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/model\darknet.py)                                                           | The provided code snippet is a part of a larger project, but it mainly contains the implementation of various modules and helper functions for a Darknet neural network. These modules include convolutional layers with batch normalization and activation functions, max pooling, reorganization, average pooling, softmax, fully connected layers, and shortcut connections. The code also includes functions for loading and saving network weights.                                                                                                                                                                  |
| [model.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/model\model.py)                                                               | The provided code snippet defines a Tech Lead module for Multi-Task Action Fusion 3D. It includes functionality for backbone models, neck modules, and detection heads. The code performs feature extraction, fusion, and inference for object detection, object classification, and action recognition tasks.                                                                                                                                                                                                                                                                                                            |
| [resnext.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/model\resnext.py)                                                           | The provided code is for implementing the ResNeXt architecture for 3D data. It includes functions for creating ResNeXt models with different depths (resnet50, resnet101, and resnet152). The code also contains the ResNeXtBottleneck and the main ResNeXt classes, which define the structure of the model. The code implements the forward pass, downsample operations, and shortcut connections. The code also includes initialization and fine-tuning parameter functions, although they are currently commented out. Overall, the code allows for the creation of ResNeXt models specifically designed for 3D data. |
| [general.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/utils\general.py)                                                           | Prompt exceeds max token limit: 5709.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [loss.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/utils\loss.py)                                                                 | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [loss_ava.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/utils\loss_ava.py)                                                         | HTTPStatus Exception: 400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [metrics.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/utils\metrics.py)                                                           | This code snippet provides functionalities for model validation metrics in object detection tasks. It includes functions for computing average precision, confusion matrix, and plotting precision-recall curves and metric-confidence curves.                                                                                                                                                                                                                                                                                                                                                                            |
| [plots.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/utils\plots.py)                                                               | HTTPStatus Exception: 429                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [scheduler.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/utils\scheduler.py)                                                       | HTTPStatus Exception: 429                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [torch_utils.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/utils\torch_utils.py)                                                   | HTTPStatus Exception: 429                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [log_dataset.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/utils\wandb_logging\log_dataset.py)                                     | The code snippet parses command line arguments, loads YAML data, and creates a dataset artifact using the WandbLogger.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [wandb_utils.py](https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2/blob/main/utils\wandb_logging\wandb_utils.py)                                     | HTTPStatus Exception: 429                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

</details>

---

## 🚀 Getting Started

### ✔️ Prerequisites

Before you begin, ensure that you have the following prerequisites installed:
> - `ℹ️ Requirement 1`
> - `ℹ️ Requirement 2`
> - `ℹ️ ...`

### 📦 Installation

1. Clone the Combined_Learner_AVA2.2_and_Deepfashion2 repository:
```sh
git clone https://github.com/JJaewon7210/Combined_Learner_AVA2.2_and_Deepfashion2
```

2. Change to the project directory:
```sh
cd Combined_Learner_AVA2.2_and_Deepfashion2
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### 🎮 Using Combined_Learner_AVA2.2_and_Deepfashion2

```sh
python main.py
```

### 🧪 Running Tests
```sh
pytest
```

---


## 🗺 Roadmap

> - [X] `ℹ️  Task 1: Implement X`
> - [ ] `ℹ️  Task 2: Refactor Y`
> - [ ] `ℹ️ ...`


---

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:
1. Fork the project repository. This creates a copy of the project on your account that you can modify without affecting the original project.
2. Clone the forked repository to your local machine using a Git client like Git or GitHub Desktop.
3. Create a new branch with a descriptive name (e.g., `new-feature-branch` or `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Make changes to the project's codebase.
5. Commit your changes to your local branch with a clear commit message that explains the changes you've made.
```sh
git commit -m 'Implemented new feature.'
```
6. Push your changes to your forked repository on GitHub using the following command
```sh
git push origin new-feature-branch
```
7. Create a new pull request to the original project repository. In the pull request, describe the changes you've made and why they're necessary.
The project maintainers will review your changes and provide feedback or merge them into the main branch.

---

## 📄 License

This project is licensed under the `ℹ️  INSERT-LICENSE-TYPE` License. See the [LICENSE](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository) file for additional info.

---

## 👏 Acknowledgments

> - `ℹ️  List any resources, contributors, inspiration, etc.`

---
