# Deep Embedded Clustering Regularization for Imbalanced Cerebral Emboli Classification Using Transcranial Doppler Ultrasound

## I) Introduction

This repository presents the code of synthetic data experiments of the submitted paper *Deep Embedded Clustering Regularization for Imbalanced Cerebral Emboli Classification Using Transcranial Doppler Ultrasound*.

## II) Configuration

Install the different libraries needed to execute the different codes:

    pip install -r requirements.txt

To be able to run the different codes, you need to start by running the following command:

- For Linux systems:

        export PYTHONPATH="${PYTHONPATH}:path_to_the_imbalanced-dec-regularization_code"



- For Windows systems:
    
        set PYTHONPATH=%PYTHONPATH%;path_to_the_imbalanced-dec-regularization_code

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

The code is structured in different folders:
- **src**: This folder contains the source codes necessary to run the different experiments. Particularly, this folder contains the subfolder *experiments* containing two files, *experiment_1.py* and *experiment_2.py*, allowing to launch the two experiments. 
The first file, *experiment_1.py*, can be launched with the option *obtain_animation_bool*, corresponding to a boolean indicating if animations of the learned 2D test embeddings over the epochs should be generated (if the case, they will be stored in a the folder results/Experiment_1). 
The second file, *experiment_2.py*, can be launched with the option "study_hyperparams", corresponding to a boolean indicating if the experiment should be done from scratch or if the pre-computed results should be used.
- **results**: This folder contains the results of the different experiments. For more information, please refer to the README file in the folder.
- **parameters_files**: This folder contains json files with the parameters of each individual model on each dataset.
- **notebooks**: This folder contains notebooks corresponding to both experiments, allowing to launch them in an interactive way.
- **figs**: This fodler contains the different figures used as illustrations in this Git repository.


## V) Examples

### A) Experiment 1

Experiment 1 can be launched using the notebook *notebooks/experiments/Experiment_1.ipynb*.

This experiment can also be launched on the command line using the file *src/experiments/experiment_1.py*:

    cd src/experiments/
    python experiment_1.py --obtain_animation_bool True
    
If *obtain_animation_bool* is set to *True* (in the notebook or in the code), animations corresponding to the learned 2D test embeddings over the epochs will generated in the folder *results/Experiment_1/*.
The results of this experiment should be similar to the following ones:
| **Dataset** | **DEC** | **MCC**      | **F1-Score** |
|-------------|---------|--------------|--------------|
|      SB     | No      | 99.90 ± 0.2   | 99.93 ± 0.13 |
|             | Yes     | 99.80 ± 0.25  | 99.87 ± 0.16 |
|      SI     | No      | 88.01 ± 3.63 | 94.83 ± 1.65 |
|             | Yes     | 93.40 ± 3.66  | 97.34 ± 1.63 |
|      NB     | No      | 96.80 ± 0.87  | 97.87 ± 0.58 |
|             | Yes     | 96.50 ± 0.98  | 97.67 ± 0.65 |
|      NI     | No      | 88.76 ± 3.55 | 95.28 ± 1.65 |
|             | Yes     | 94.11 ± 1.34 | 97.81 ± 0.44 |

### B) Experiment 2

Experiment 2 can be launched using the notebook *notebooks/experiments/Experiment_2.ipynb*.

This experiment can also be launched on the command line using the file *src/experiments/experiment_2.py*:

    cd src/experiments/
    python experiment_2.py --study_hyperparams True
    
The results (for the notebook or the python file), will be stored in *results/Experiment_2/. If they are launched with the option *study_hyperparams* set to *False*, then pre-computed results will be used. 
The results of this experiment should be similar to the following ones:

![alt text](https://github.com/gitanonymoussubmussion/imbalanced_dec_regularization/blob/main/figs/Results/Experiment_2/MatrixValues.png)
