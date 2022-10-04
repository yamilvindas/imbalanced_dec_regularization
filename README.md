# Deep Embedded Clustering Regularization for Imbalanced Cerebral Emboli Classification Using Transcranial Doppler Ultrasound

## I) Introduction

This repository presents the code of synthetic data experiments of the submitted paper *Deep Embedded Clustering Regularization for Imbalanced Cerebral Emboli Classification Using Transcranial Doppler Ultrasound*.

## II) Configuration

Install the different libraries needed to execute the different codes:

    pip install -r requirements.txt

To be able to run the different codes, you need to start by running the following command:

    export PYTHONPATH="${PYTHONPATH}:path_to_the_imbalanced-dec-regularization_code"

## III) Proposed Method

![alt text](https://github.com/gitanonymoussubmussion/imbalanced-dec-regularization/blob/main/figs/Method/GlobalPipeline.jpg)

Our propsed method decomposes a classification in two parts: an **encoder** and a **classifier**. Then, we do (simulatenously) unsupervised clustering by applying DEC [(Xie et al., 2016)](https://arxiv.org/abs/1511.06335) to the encoder's latent space, and supervised trained classification by optimizing the *Cross-Entropy* loss betwwen the output of the classifier (which uses the encoder's latent space) and the true labels of the samples.

We used two models taking different inputs from [(Vindas et al., 2022)](https://www.mlforhc.org/s/43-Paper-43_CameraReady.pdf): (1) a 2D CNN model taking as input a time-frequency representation, and (2) a 1D CNN-Transformer model taking as input a raw signal. The model architectures are described in the following figures:

| ![](https://github.com/gitanonymoussubmussion/imbalanced-dec-regularization/blob/main/figs/Method/2DCNN.jpg) | 
|:--:| 
| *2D CNN model* |

| ![](https://github.com/gitanonymoussubmussion/imbalanced-dec-regularization/blob/main/figs/Method/1DCNN_Transformer.jpg) | 
|:--:| 
| *1D CNN-Transformer model* |

For privacy reasons, the cerebral emboli dataset used in the final experiment of the paper is not available here. Only the experiments on four synthetic datasets are available in this repository. The four used datasets are the following:

| ![](https://github.com/gitanonymoussubmussion/imbalanced-dec-regularization/blob/main/figs/Dataset/SeparableBalanced.png) | 
|:--:| 
| *Balanced and linearly separable* |

| ![](https://github.com/gitanonymoussubmussion/imbalanced-dec-regularization/blob/main/figs/Dataset/NotSeparableBalanced.png) | 
|:--:| 
| *Balanced and nonlinearly separable* |


| ![](https://github.com/gitanonymoussubmussion/imbalanced-dec-regularization/blob/main/figs/Dataset/SeparableUnbalanced.png) | 
|:--:| 
| *Imbalanced and nonlinearly separable* |


| ![](https://github.com/gitanonymoussubmussion/imbalanced-dec-regularization/blob/main/figs/Dataset/NotSeparableUnbalanced.png) | 
|:--:| 
| *Imbalanced and nonlinearly separable* |


The model used for these datasets is composed of one fully connected (FC) linear layer followed by a softplus activation function (encoder). Then, an FC layer is applied, followed by a softmax function to do classification (classifier). 

## IV) Code structure

Todo

## V) Examples

Todo (put results tables and figures)
