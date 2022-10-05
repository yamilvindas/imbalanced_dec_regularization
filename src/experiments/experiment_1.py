#!/usr/bin/env python3
"""
    Executres the Experiment 1 of the submitted paper. This experiment consist
    on comparing DEC regularization on four synthetic 3D datasets composed of
    three classes:
        - D_SB: Balanced with all the classes being linearly separable between
        each other.
        - D_SI: Imalanced with all the classes being linearly separable between
        each other.
        - D_NB: Balanced with two nonlinealry separable classes.
        - D_NI: Imalanced with two nonlinealry separable classes.

    Options:
    --------
    --obtain_animation_bool: str (bool)
        Boolean to create (or not) animations showing the evolution of the test embeddings over the epochs (for one fixed repetition)
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from src.experiments.train_single_model import TrainSingleModel
from src.utils.tools import plot_points_by_class_2D, plot_points_by_class_3D, plot_animation, get_mean_metrics

class Experiment1(object):
    def __init__(self, obtain_animation_bool):
        """
            Constructor

            Arguments:
            ----------
            - obtain_animation_bool: bool
                Boolean to create (or not) animations showing the evolution of
                the test embeddings over the epochs (for one fixed repetition)
        """
        # Boolean for the creation of the animations
        self.obtain_animation_bool = obtain_animation_bool

        # Parameters of the experiment variable
        self.sub_exps_params = {
                                    'SB No DEC': {},
                                    'SB DEC': {},
                                    'SI No DEC': {},
                                    'SI DEC': {},
                                    'NB No DEC': {},
                                    'NB DEC': {},
                                    'NI No DEC': {},
                                    'NI DEC': {},
                                }
        # Dataset to use
        self.sub_exps_params['SB DEC']['dataset_type'] = 'balanced_separated'
        self.sub_exps_params['SB No DEC']['dataset_type'] = 'balanced_separated'
        self.sub_exps_params['SI DEC']['dataset_type'] = 'imbalanced_separated'
        self.sub_exps_params['SI No DEC']['dataset_type'] = 'imbalanced_separated'
        self.sub_exps_params['NB DEC']['dataset_type'] = 'balanced_mixed'
        self.sub_exps_params['NB No DEC']['dataset_type'] = 'balanced_mixed'
        self.sub_exps_params['NI DEC']['dataset_type'] = 'imbalanced_mixed'
        self.sub_exps_params['NI No DEC']['dataset_type'] = 'imbalanced_mixed'

        # Other parameters
        nb_repetitions = 10
        batch_size = 32
        for sub_exp_name in self.sub_exps_params:
            # Use DEC
            if ('No DEC' in sub_exp_name):
                self.sub_exps_params[sub_exp_name]['use_dec'] = False
            else:
                self.sub_exps_params[sub_exp_name]['use_dec'] = True
            # Number of repetitions
            self.sub_exps_params[sub_exp_name]['nb_repetitions'] = nb_repetitions
            # Batch size
            self.sub_exps_params[sub_exp_name]['batch_size'] = batch_size

            # DEC hyper-parameters and training parameters
            if ('SB' in sub_exp_name) or ('NI' in sub_exp_name):
                self.sub_exps_params[sub_exp_name]['importance_dec'] = 1
                self.sub_exps_params[sub_exp_name]['epoch_init_dec_loss'] = 5
                self.sub_exps_params[sub_exp_name]['nb_epochs'] = 50
                self.sub_exps_params[sub_exp_name]['lr'] = 5e-2
            if ('SI' in sub_exp_name):
                self.sub_exps_params[sub_exp_name]['importance_dec'] = 10
                self.sub_exps_params[sub_exp_name]['epoch_init_dec_loss'] = 1
                self.sub_exps_params[sub_exp_name]['nb_epochs'] = 50
                self.sub_exps_params[sub_exp_name]['lr'] = 5e-3
            if ('NB' in sub_exp_name):
                self.sub_exps_params[sub_exp_name]['importance_dec'] = 5e-1
                self.sub_exps_params[sub_exp_name]['epoch_init_dec_loss'] = 1
                self.sub_exps_params[sub_exp_name]['nb_epochs'] = 100
                self.sub_exps_params[sub_exp_name]['lr'] = 1e-2

    def run(self):
        """
            Run the full experiment.
        """
        metrics_experiment = {sub_exp_name:{} for sub_exp_name in self.sub_exps_params}
        for sub_exp_name in self.sub_exps_params:
            # Fixing the random seed
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

            # Creation of an instance of the sub-experiment
            single_train_exp = TrainSingleModel(self.sub_exps_params[sub_exp_name])

            # Plotting the generated data
            plot_points_by_class_3D(single_train_exp.X_data, single_train_exp.Y_data, centroids=None, close_fig=False)

            # Training the model
            single_train_exp.repeatedHoldout()

            # Print the mean metrics
            mean_last_mcc, std_last_mcc = get_mean_metrics(single_train_exp.list_last_epoch_metrics, metric_type='MCC')
            mean_last_f1_score, std_last_f1_score = get_mean_metrics(single_train_exp.list_last_epoch_metrics, metric_type='F1-Score')
            mean_last_accuracy, std_last_accuracy = get_mean_metrics(single_train_exp.list_last_epoch_metrics, metric_type='Accuracy')

            # Adding the metrics to the dictionary of the metrics of all the experiments
            metrics_experiment[sub_exp_name]['MCC'] = '{} \u00B1 {}'.format(round(mean_last_mcc, 2), round(std_last_mcc, 2))
            metrics_experiment[sub_exp_name]['F1-Score'] = '{} \u00B1 {}'.format(round(mean_last_f1_score, 2), round(std_last_f1_score, 2))
            metrics_experiment[sub_exp_name]['Accuracy'] = '{} \u00B1 {}'.format(round(mean_last_accuracy, 2), round(std_last_accuracy, 2))

            # Getting the encodings and cluster centers for a given repetition
            rep_to_use = 0
            encodings_by_epochs = single_train_exp.list_encodings_by_epochs[rep_to_use]
            cluster_centers_by_epoch = single_train_exp.list_cluster_centers_by_epoch[rep_to_use]

            # Plotting the 2D test encodings at the last epoch
            epoch = len(encodings_by_epochs['Test']) - 1
            test_encodings = encodings_by_epochs['Test'][epoch]['Encodings']
            X_data_samples = np.array([ test_encodings[i] for i in range(len(test_encodings)) ])
            Y_data_samples = np.array(encodings_by_epochs['Test'][epoch]['TrueLabels'])
            if (epoch not in cluster_centers_by_epoch):
                plot_points_by_class_2D(X_data_samples, Y_data_samples, centroids=None, show_fig=True, save_fig=False)
            else:
                plot_points_by_class_2D(X_data_samples, Y_data_samples, centroids=cluster_centers_by_epoch[epoch], show_fig=True, save_fig=False)

            # Plotting the animation with the original labels
            if (self.obtain_animation_bool):
                for epoch in range(single_train_exp.nb_epochs):
                    test_encodings = encodings_by_epochs['Test'][epoch]['Encodings']
                    X_data_samples = np.array([test_encodings[i] for i in range(len(test_encodings)) ])
                    Y_data_samples = np.array(encodings_by_epochs['Test'][epoch]['TrueLabels'])
                    if (epoch not in cluster_centers_by_epoch):
                        plot_points_by_class_2D(X_data_samples, Y_data_samples, centroids=None, show_fig=False, save_fig=True, file_name_fig='tmp_img_{}_epoch-{}.png'.format(sub_exp_name, epoch))
                    else:
                        plot_points_by_class_2D(X_data_samples, Y_data_samples, centroids=cluster_centers_by_epoch[epoch], show_fig=False, save_fig=True, file_name_fig='tmp_img_{}_epoch-{}.png'.format(sub_exp_name, epoch))

                # Plotting the animation
                files_image_list = ['../../results/.tmp_images_for_gif/tmp_img_{}_epoch-{}.png'.format(sub_exp_name, i) for i in range(single_train_exp.nb_epochs)]
                if (not os.path.isdir('../../results/Experiment_1/')):
                    os.mkdir('../../results/Experiment_1/')
                animation_file_name = '../../results/Experiment_1/TestEmbeddings_{}.mp4'.format(sub_exp_name)
                plot_animation(files_image_list, animation_file_name)

            # Erasing all the images in the tmpImagesForGIF folder
            for file in os.listdir("../../results/.tmp_images_for_gif/"):
                os.remove("../../results/.tmp_images_for_gif/"+file)

        # Plotting the results in a table format
        results_df = pd.DataFrame(metrics_experiment).T
        print(results_df)


#==============================================================================#
#==============================================================================#
def main():
    print("\n\n==================== Beginning of the experiment ====================\n\n")
    #==========================================================================#
    # Fixing the random seed
    random.seed(42) # For reproducibility purposes
    np.random.seed(42) # For reproducibility purposes
    torch.manual_seed(42) # For reproducibility purposes

    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--obtain_animation_bool', default="False", help="Boolean to create (or not) animations showing the evolution of the test embeddings over the epochs (for one fixed repetition)", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    obtain_animation_bool = args['obtain_animation_bool']
    if (obtain_animation_bool.lower() == 'true'):
        obtain_animation_bool = True
    else:
        obtain_animation_bool = False

    #==========================================================================#
    # Creating an instance of the experiment
    exp_1 = Experiment1(obtain_animation_bool)

    # Running the experiment
    exp_1.run()

if __name__=="__main__":
    main()
