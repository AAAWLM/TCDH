## TCDH
The source code of the **T**riplet-**C**onstrained **D**eep **H**ashing (TCDH) framework.

## Paper
**Triplet-Constrained Deep Hashing for Chest X-Ray Image Search**

Linmin Wang, Qianqian Wang, Xiaochuan Wang, Yunling Ma, Limei Zhang, Mingxia Liu

## Dataset
We used the following dataset:

-COVIDx (Can be downloaded [here](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2?select=competition_test))
Note that this dataset is not static but constantly changes due to the new COVID images being added. The set of training and testing images we used can be found in train_split.txt and test_COVIDx4.txt, respectively.

## Dependencies
TCDH needs the following dependencies:

- python 3.8.5
- PIL == 9.2.0
- torch == 1.13.0
- numpy == 1.23.3
- torchvision == 0.14.0

## Structure
    - `./training.py`: The main functions for TCDH.
    - `./read_dataset.py`: Data preparation for COVIDx.
    - `./model.py`: The model used in TCDH.
    - `./Triplet_loss.py`: This is Triplet loss function.
    - `./PK_sampler.py`: This is sampling strategy of Triplet loss.

