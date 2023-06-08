# Noisy Adversarial Representation Learning for Effective and Efficient Image Obfuscation

# Dependencies
- python 3.8
- CUDA 10.2
- pytorch 1.11.0
    - torchvision 0.12.0
    - cudatoolkit 10.2
- `$ pip install -r requirements.txt`

# Dataset
## Fairface
- paper : https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf
- official repo: https://github.com/dchen236/FairFace

### Setup
1. Download data from [official repo](https://github.com/dchen236/FairFace#data)
    1. Images: `padding=0.25`
    2. Labels: `Train`, `Val`
2. Unzip image zip and directory structure would be
```
fairface/
|-- train/
    |-- 1.jpg
    |-- 2.jpg
    |-- ...
|-- val/
    |-- 1.jpg
    |-- 2.jpg
    |-- ...
|-- fairface_label_train.csv
|-- fairface_label_val.csv
```
3. Specify `DATA_ROOT` in `dataset/fairface.py`.

## CelebA, CIFAR10
- `torchvision` implementation used.
- Specify `DOWNLOAD` and `DATA_ROOT` in  `dataset/{celeba, cifar10}.py`.


# Training
The training consists of three steps.
1. Train obfuscator 
2. Train adversary classifier to verify adversary task defense.
3. Train adversary reconstructor to verify adversary reconstruction defense.