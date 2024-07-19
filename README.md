# RED
This repository contains code for our under-review paper "RED: Efficiently Boosting Ensemble Robustness via Random Sampling Inference".

# Dependencies
We were using PyTorch 1.10.0 for all the experiments. You may want to install other versions of PyTorch according to the cuda version of your computer/server.
The code is run and tested on the Artemis HPC server, 4090 lab-cluster server, and NCI server with multiple GPUs. Running on a single GPU may need adjustments.

# Data and pre-trained models
We used the pre-trained models for the [PyTorch package](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html). We used the CIFAR-10 and Tiny-ImageNet  datasets to fine-train and evaluate our proposed models and the baselines. 

# Usage
Examples of training and evaluation scripts can be found in `train.py` and `train_hyper.py`.
