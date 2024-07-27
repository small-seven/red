# RED: Efficiently Boosting Ensemble Robustness via Random Sampling Inference
This repository contains code for our under-review paper "RED: Efficiently Boosting Ensemble Robustness via Random Sampling Inference".

# Dependencies
We were using PyTorch 1.10.0 for all the experiments. You can install other versions of PyTorch according to the cuda version of your computer/server.
The code is run and tested on the Artemis HPC server, 4090 lab-cluster server, and NCI server with multiple GPUs. Running on a single GPU may need adjustments. The full dependencies are shown in `requirements.txt`.

# Data and pre-trained models
We used the pre-trained models for the [PyTorch package](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html). We used the CIFAR-10 and Tiny-ImageNet  datasets to fine-train and evaluate our proposed models and the baselines. 

# Usage
Examples of training and evaluation scripts can be found in `train.py`:
```
python train.py --train_mode=red --train_batch_size=256 --epochs=120 --num_models=3 --lambda_1=10.0 --lambda_2=10.0 --plus_adv --proj_dir="./"
```
, `train_hyper.py`:
```
python train_hyper.py --train_mode=hyper --train_batch_size=256 --epochs=120 --num_models=3 --lambda_1=10.0 --lambda_2=10.0 --plus_adv --embedding=128 --proj_dir="./"
```
and `test.py`:
```
python test.py --train_mode=red --num_models=3 --lambda_1=10.0 --lambda_2=10.0 --proj_dir="./" --attack_type=pgd --red_test --eval
```
