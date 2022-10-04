#!/usr/bin/env python3
"""
    This codes implements different functions allowing to create different
    synthetic datasets of 3D points distributed in 3 classes
"""
import numpy as np
import torch
from torch.utils.data import Dataset

def generate_synthetic_data(type_data_generate='imbalanced_mixed'):
    """
        Generates synthetic data composed of 3D points distributed in three
        classes.

        Arguments:
        ----------
        type_data_generate: str
            Type of data points to generate. There are four options:
                - balanced_separated
                - imbalanced_separated
                - balanced_mixed
                - imbalanced_mixed

        Returns:
        --------
        X_data: np.array
            Array of 3D data points.
        Y_data: np.array
            Labels of the 3D data points.
    """
    # Well separated and balanced dataset
    if (type_data_generate.lower() == 'balanced_separated'):
        centre_1, dispersion_1, nb_samples_cluster_1 = -2, 0.5, 500
        centre_2, dispersion_2, nb_samples_cluster_2 = 0, 0.5, 500
        centre_3, dispersion_3, nb_samples_cluster_3 = 2, 0.5, 500
    # Well separated and imbalanced dataset
    if (type_data_generate.lower() == 'imbalanced_separated'):
        centre_1, dispersion_1, nb_samples_cluster_1 = -2, 0.5, 1000
        centre_2, dispersion_2, nb_samples_cluster_2 = 0, 0.5, 200
        centre_3, dispersion_3, nb_samples_cluster_3 = 2, 0.5, 4000
    # Bad separated and balanced dataset
    if (type_data_generate.lower() == 'balanced_mixed'):
        centre_1, dispersion_1, nb_samples_cluster_1 = -1, 0.5, 500
        centre_2, dispersion_2, nb_samples_cluster_2 = 0, 0.45, 500
        centre_3, dispersion_3, nb_samples_cluster_3 = 2.5, 0.5, 500
    # Bad separated and unbalances dataset
    if (type_data_generate.lower() == 'imbalanced_mixed'):
        centre_1, dispersion_1, nb_samples_cluster_1 = -1, 0.5, 1000
        centre_2, dispersion_2, nb_samples_cluster_2 = 0, 0.45, 200
        centre_3, dispersion_3, nb_samples_cluster_3 = 2.5, 0.5, 4000

    # Generation of the clusters
    # First cluster
    x1 = np.random.normal(loc=centre_1, scale=dispersion_1, size=nb_samples_cluster_1)
    y1 = np.random.normal(loc=centre_1, scale=dispersion_1, size=nb_samples_cluster_1)
    z1 = np.random.normal(loc=centre_1, scale=dispersion_1, size=nb_samples_cluster_1)
    cluster_1 = np.array([[x1[i], y1[i], z1[i]] for i in range(nb_samples_cluster_1)])

    # Second cluster
    x2 = np.random.normal(loc=centre_2, scale=dispersion_2, size=nb_samples_cluster_2)
    y2 = np.random.normal(loc=centre_2, scale=dispersion_2, size=nb_samples_cluster_2)
    z2 = np.random.normal(loc=centre_2, scale=dispersion_2, size=nb_samples_cluster_2)
    cluster_2 = np.array([[x2[i], y2[i], z2[i]] for i in range(nb_samples_cluster_2)])

    # Third cluster

    x3 = np.random.normal(loc=centre_3, scale=dispersion_3, size=nb_samples_cluster_3)
    y3 = np.random.normal(loc=centre_3, scale=dispersion_3, size=nb_samples_cluster_3)
    z3 = np.random.normal(loc=centre_3, scale=dispersion_3, size=nb_samples_cluster_3)
    cluster_3 = np.array([[x3[i], y3[i], z3[i]] for i in range(nb_samples_cluster_3)])

    # Getting the final data
    X_data = np.concatenate((cluster_1, cluster_2, cluster_3))
    Y_data = np.array(
                        [0 for i in range(cluster_1.shape[0])]\
                      + [1 for i in range(cluster_2.shape[0])]\
                      + [2 for i in range(cluster_3.shape[0])]
                     )

    return X_data, Y_data

# Creating the new dataste class
class SyntheticDataset(Dataset):
    def __init__(self, X_data, Y_data):
        super().__init__()
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, i):
        # Data point
        sample_data = self.X_data[i]

        # Label
        sample_label = self.Y_data[i]

        return torch.from_numpy(sample_data), sample_label
