# Object-aware Gaze Target Detection
[![arXiv](https://img.shields.io/badge/arXiv-2307.09662-00ff00.svg)](https://arxiv.org/abs/2307.09662)

Official repo of the paper "Object-aware Gaze Target Detection" (ICCV 2023).

![Method](./assets/method.png)

## Description

This repo contains all the code to train and evaluate our method.
The code is based on [PyTorch Lightning](https://www.lightning.ai/) and [Hydra](https://hydra.cc/).

Please follow the instructions below to install dependencies and run the code.
We provide configurations to train the model on GazeFollow and VideoAttentionTarget, and you can easily tune them by looking at the parameters of each module in the [configs/](configs/) folder.

## Prerequisites
### Environment and dependencies
We provide a pip requirements file to install all the dependencies.
We recommend to use a conda environment to install the dependencies.

```bash
# Clone project and submodules
git clone --recursive https://github.com/francescotonini/object-aware-gaze-target-detection.git
cd object-aware-gaze-target-detection

# Create conda environment
conda create -n object-aware-gaze-target-detection python=3.9
conda activate object-aware-gaze-target-detection

# Install requirements
pip install -r requirements.txt
```

(optional) Setup wandb
```bash
cp .env.example .env

# Add token to .env
```

### Dataset preprocessing
The code expects that the datasets are placed under the [data/](data/) folder.
You can change this by modifying the `data_dir` parameter in the configuration files.

```bash
cat <<EOT >> configs/local/default.yaml
# @package _global_

paths:
  data_dir: "{PATH TO DATASETS}"
EOT
```

The implementation requires both object and faces annotations.
Therefore, you need run the following script to extract face and object annotations.

```bash
# GazeFollow
python scripts/gazefollow_get_aux_faces.py --dataset_path /path/to/gazefollow --subset train
python scripts/gazefollow_get_aux_faces.py --dataset_path /path/to/gazefollow --subset test
python scripts/gazefollow_get_objects.py --dataset_path /path/to/gazefollow --subset train
python scripts/gazefollow_get_objects.py --dataset_path /path/to/gazefollow --subset test

# VideoAttentionTarget
python scripts/videoattentiontarget_get_aux_faces.py --dataset_path /path/to/videoattentiontarget --subset train
python scripts/videoattentiontarget_get_aux_faces.py --dataset_path /path/to/videoattentiontarget --subset test
python scripts/videoattentiontarget_get_objects.py --dataset_path /path/to/videoattentiontarget --subset train
python scripts/videoattentiontarget_get_objects.py --dataset_path /path/to/videoattentiontarget --subset test
```

## Training
We provide configuration to train on GazeFollow and VideoAttentionTarget (see [configs/experiment/](configs/experiment/)).
First, you need to pretrain the method for object detection only.

```bash
python src/train.py experiment=gotd_gazefollow_pretrain_od
```

The pretraining is useful to initialize the object detection head of the model for face recognition.
Then, you can train the model on GazeFollow or VideoAttentionTarget.

```bash
# GazeFollow
python src/train.py experiment=gotd_gazefollow model.net_pretraining={URL/PATH TO GAZEFOLLOW OD PRETRAINING}

# VideoAttentionTarget
python src/train.py experiment=gotd_videoattentiontarget model.net_pretraining={URL/PATH TO GAZEFOLLOW TRAINED MODEL}
```

## Evaluation
The configuration files are useful also when evaluating the model.

```bash
# GazeFollow
python src/eval.py experiment=gotd_gazefollow ckpt_path={PATH TO CHECKPOINT}

# VideoAttentionTarget
python src/eval.py experiment=gotd_videoattentiontarget ckpt_path={PATH TO CHECKPOINT}
```

## Acknowledgements
This code is based on [PyTorch Lightning](https://www.lightning.ai/), [Hydra](https://hydra.cc/), and the official DETR implementation.

## Cite us
```
@InProceedings{tonini2023object,
  title={Object-aware Gaze Target Detection},
  author={Tonini, Francesco and Dall'Asen, Nicola and Beyan, Cigdem and Ricci, Elisa},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2023}
}
```

