#!/usr/bin/env python3
"""
    Executres the Experiment 2 of the submitted paper. This experiment consist
    on studying the influence of two hyperparameters of the DEC regularization
    term: gamma (importance of VAT) and e_init (epoch from which we activate
    the DEC loss).
    All the sub-experiments are done in a 3D synthetic dataset with three classes,
    two of them being nonlinealry separable

    Options:
    --------
    study_hyperparams: str (bool)
        Boolean indicating if we want to start an experiment from scratch or
        use the results of a pre-computed experiment
"""
import os
import argparse
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.experiments.train_single_model import TrainSingleModel
from src.utils.tools import get_mean_metrics

class Experiment2(object):
    def __init__(self, study_hyperparams):
        """
            Constructor

            Arguments:
            ----------
            study_hyperparams: bool
                Boolean indicating if we want to start an experiment from scratch or
                use the results of a pre-computed experiment
        """
        # Boolean to start the experiment from scratch
        self.study_hyperparams = study_hyperparams

        # Parameters of the experiment variable
        self.base_parameters_exp = {}

        # Dataset to use
        self.base_parameters_exp['dataset_type'] = 'imbalanced_mixed'

        # Use DEC
        self.base_parameters_exp['use_dec'] = True

        # Number of repetitions
        #self.base_parameters_exp['nb_repetitions'] = 10
        self.base_parameters_exp['nb_repetitions'] = 2

        # Batch size
        self.base_parameters_exp['batch_size'] = 32

        # Epochs
        self.base_parameters_exp['nb_epochs'] = 50

        # Learning rate
        self.base_parameters_exp['lr'] = 5e-2

        # Hyperparameters to explore
        # e_init
        # self.vals_epoch_init_dec_loss = [0, 1, 5, 10, 20, int(self.base_parameters_exp['nb_epochs']/2.)]
        self.vals_epoch_init_dec_loss = [0, 1]
        # Gamma
        # self.vals_importance_dec = [10, 7, 5, 3, 1, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001, 0.0001]
        self.vals_importance_dec = [10, 7]

        # Axis tick size
        self.axisTickSize = 30
        # Legend fontsize
        self.legendFontSize = 25


    def run(self):
        """
            Run the full experiment.
        """
        # Variable to store the results
        self.results_combination_hyperparams = {}

        if (self.study_hyperparams):
            print("=======> Starting the study of the influence of the hyperparameters <=======")
            # Exploring the different combinations
            for importance_dec in self.vals_importance_dec:
                for epoch_init_dec_loss in self.vals_epoch_init_dec_loss:
                    print("\t\t=======> Using (importance_dec, epoch_init_dec_loss) = ({}, {})".format(importance_dec, epoch_init_dec_loss))
                    # Fixing the random seed
                    random.seed(42)
                    np.random.seed(42)
                    torch.manual_seed(42)

                    # Adding the DEC hyperparameters to the list of parameters of the experiment
                    self.base_parameters_exp['importance_dec'] = importance_dec
                    self.base_parameters_exp['epoch_init_dec_loss'] = epoch_init_dec_loss

                    # Creation of an instance of the sub-experiment
                    single_train_exp = TrainSingleModel(self.base_parameters_exp)

                    # Training the model
                    single_train_exp.repeatedHoldout()

                    # Storing the results
                    self.results_combination_hyperparams[(importance_dec, epoch_init_dec_loss)] = {
                                                                                                    'LossVals_PerRep': single_train_exp.list_loss_values,
                                                                                                    'LastEpochMetrics_PerRep': single_train_exp.list_last_epoch_metrics
                                                                                               }

            # Saving the final results
            if (not os.path.isdir('../../results/Experiment_2/')):
                os.mkdir('../../results/Experiment_2/')
            inc = 0
            res_file_name = '../../results/Experiment_2/res_exp_hyperparams_'
            os.path.exists(res_file_name + str(inc) + '.pth')
            while (os.path.exists(res_file_name + str(inc) + '.pth')):
                inc += 1
            res_file_name = res_file_name + str(inc) + '.pth'
            with open(res_file_name, "wb") as fp:
                pickle.dump(self.results_combination_hyperparams, fp)
            print("=======> Finishing the study of the influence of the hyperparameters <=======")
        else:
            print("=======> Starting the LOADING study of the influence of the hyperparameters <=======")
            # Opening the results file
            res_file_name = '../../results/Experiment_2/res_exp_hyperparams_0.pth'
            with open(res_file_name, "rb") as pf:
                self.results_combination_hyperparams = pickle.load(pf)
            print("=======> Finishing the LOADING study of the influence of the hyperparameters <=======")

    def plotResultsMatrix(self):
        # Plotting a matrix to show the MCC based on importance_dec and e_init (as in Lueks et al. paper about
        # local quality)
        # Getting the MCC values
        mcc_matrix = np.zeros((len(self.vals_epoch_init_dec_loss), len(self.vals_importance_dec)))
        for epoch_init_dec_loss_ID in range(len(self.vals_epoch_init_dec_loss)):
            for importance_dec_ID in range(len(self.vals_importance_dec)):
                # Getting the values of epoch_init_dec_loss and importance_dec
                epoch_init_dec_loss = self.vals_epoch_init_dec_loss[epoch_init_dec_loss_ID]
                importance_dec = self.vals_importance_dec[importance_dec_ID]

                # Getting the results for that combination of hyperparams
                results_one_combination = self.results_combination_hyperparams[(importance_dec, epoch_init_dec_loss)]

                # Getting the metrics for that combination
                print("For combination (e_init, \u03B3) = ({}, {}) we have: ".format(importance_dec, epoch_init_dec_loss))
                mean_last_mcc, std_last_mcc = get_mean_metrics(results_one_combination['LastEpochMetrics_PerRep'], metric_type='MCC')

                # Adding the values to the MCC dict
                mcc_matrix[epoch_init_dec_loss_ID, importance_dec_ID] = mean_last_mcc


        # Doing the plot
        fig, ax = plt.subplots()
        psm = ax.pcolormesh(mcc_matrix, cmap='gist_gray')
        cbar = fig.colorbar(psm, ax=ax)
        plt.xlabel("\u03B3")
        plt.ylabel("e\N{LATIN SUBSCRIPT SMALL LETTER I}\N{LATIN SUBSCRIPT SMALL LETTER N}\N{LATIN SUBSCRIPT SMALL LETTER I}\N{LATIN SUBSCRIPT SMALL LETTER T} (in epochs)")
        plt.xticks([i for i in range(len(self.vals_importance_dec))], self.vals_importance_dec, fontsize=self.axisTickSize)
        plt.yticks([i for i in range(len(self.vals_epoch_init_dec_loss))], self.vals_epoch_init_dec_loss, fontsize=self.axisTickSize)
        plt.gca().invert_yaxis()
        plt.legend(fontsize=self.legendFontSize)
        cbar.ax.tick_params(labelsize=self.legendFontSize)
        plt.show()

    def plotMccFunctionGamma(self):
        # Plotting a curve of the MCC as a function of importance_dec
        mcc_for_fixed_importance_dec = {}
        for epoch_init_dec_loss in self.vals_epoch_init_dec_loss:
            for importance_dec in self.vals_importance_dec:
                # Getting the results for that combination of hyperparams
                results_one_combination = self.results_combination_hyperparams[(importance_dec, epoch_init_dec_loss)]

                # Getting the metrics for that combination
                print("For combination (e_init, \u03B3) = ({}, {}) we have: ".format(importance_dec, epoch_init_dec_loss))
                mean_last_mcc, std_last_mcc = get_mean_metrics(results_one_combination['LastEpochMetrics_PerRep'], metric_type='MCC')


                # Plotting the curves
                if (epoch_init_dec_loss) not in mcc_for_fixed_importance_dec:
                    mcc_for_fixed_importance_dec[epoch_init_dec_loss] = {
                                                                            'MCCMeans': [mean_last_mcc],
                                                                            'MCCStds': [std_last_mcc],
                                                                            'ImportanceDEC': [importance_dec]
                                                                        }
                else:
                    mcc_for_fixed_importance_dec[epoch_init_dec_loss]['MCCMeans'].append(mean_last_mcc)
                    mcc_for_fixed_importance_dec[epoch_init_dec_loss]['MCCStds'].append(std_last_mcc)
                    mcc_for_fixed_importance_dec[epoch_init_dec_loss]['ImportanceDEC'].append(importance_dec)

        # Doing the plot
        plt.figure()
        for epoch_init_dec_loss in mcc_for_fixed_importance_dec:
            x = mcc_for_fixed_importance_dec[epoch_init_dec_loss]['ImportanceDEC']
            y = mcc_for_fixed_importance_dec[epoch_init_dec_loss]['MCCMeans']
            yerr = mcc_for_fixed_importance_dec[epoch_init_dec_loss]['MCCStds']
            e_init_str = "e\N{LATIN SUBSCRIPT SMALL LETTER I}\N{LATIN SUBSCRIPT SMALL LETTER N}\N{LATIN SUBSCRIPT SMALL LETTER I}\N{LATIN SUBSCRIPT SMALL LETTER T}"
            plt.errorbar(x=x, y=y, yerr=yerr, label="{} = {}".format(e_init_str, epoch_init_dec_loss))
        plt.xticks(fontsize=self.axisTickSize)
        plt.yticks(fontsize=self.axisTickSize)
        plt.title("MCC as a function of importance_dec")
        plt.xlabel("\u03B3")
        plt.ylabel("MCC")
        plt.legend(fontsize=self.legendFontSize)
        plt.show()


    def boxPlotsByImportanceDEC(
                                    self,
                                    sub_exps_colors=None,
                                ):
        # Getting the repetitions of the last epoch
        last_epoch_metrics_dict = {}
        for importance_dec in self.vals_importance_dec:
            last_epoch_metrics_dict[importance_dec] = {}
            for epoch_init_dec_loss in self.vals_epoch_init_dec_loss:
                # Getting the last epoch metrics for that combination of hyperparams
                repetitions_last_epoch_metrics = self.results_combination_hyperparams[(importance_dec, epoch_init_dec_loss)]['LastEpochMetrics_PerRep']

                # Getting the metrics
                # MCC
                mcc_dict = {
                                'Train': [repetitions_last_epoch_metrics[i]['Train']['MCC'] for i in range(self.base_parameters_exp['nb_repetitions'])],
                                'Test': [repetitions_last_epoch_metrics[i]['Test']['MCC'] for i in range(self.base_parameters_exp['nb_repetitions'])]
                            }
                # F1-Score
                f1_score_dict = {
                                'Train': [repetitions_last_epoch_metrics[i]['Train']['F1-Score'] for i in range(self.base_parameters_exp['nb_repetitions'])],
                                'Test': [repetitions_last_epoch_metrics[i]['Test']['F1-Score'] for i in range(self.base_parameters_exp['nb_repetitions'])]
                            }
                # Accuracy
                acc_dict = {
                                'Train': [repetitions_last_epoch_metrics[i]['Train']['Accuracy'] for i in range(self.base_parameters_exp['nb_repetitions'])],
                                'Test': [repetitions_last_epoch_metrics[i]['Test']['Accuracy'] for i in range(self.base_parameters_exp['nb_repetitions'])]
                            }

                # Adding the metrics to the dictionary of the metrics of the last epoch
                last_epoch_metrics_dict[importance_dec][epoch_init_dec_loss] = {
                                                                                    'MCC': mcc_dict,
                                                                                    'F1Score': f1_score_dict,
                                                                                    'Accuracy': acc_dict
                                                                                }

        # Boxplot parameters
        exp_box_plot_offset_step = 5
        sub_exp_box_plot_offset_step = 0.7
        boxprops = dict(linestyle='--', linewidth=2.5)
        whiskerprops = dict(linestyle='-', linewidth=2.5)
        flierprops = dict(marker='o', markersize=12)
        medianprops = dict(linestyle='-', linewidth=2.0)
        meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')

        # Creating the plotting boxes
        fig = plt.figure()
        exp_box_plot_offset = 0
        epoch_to_use = -1
        for importance_dec in last_epoch_metrics_dict:
            sub_exp_box_plot_offset = exp_box_plot_offset
            is_named_position = False
            for epoch_init_dec_loss in last_epoch_metrics_dict[importance_dec]:
                if (sub_exps_colors is None):
                    c = 'xkcd:blue'
                else:
                    c = sub_exps_colors[epoch_init_dec_loss]
                plt.xlabel("\u03B3")
                plt.ylabel("MCC in %")
                plt.xticks(fontsize=self.axisTickSize)
                plt.yticks(fontsize=self.axisTickSize)
                plt.title("MCC as a function of importance_dec")
                data_bp = last_epoch_metrics_dict[importance_dec][epoch_init_dec_loss]['MCC']['Test']
                if (is_named_position == False):
                    labelExp = importance_dec
                    is_named_position = True
                else:
                    labelExp = ""
                bpl = plt.boxplot(
                                    data_bp,\
                                    positions=[sub_exp_box_plot_offset],\
                                    sym='o',\
                                    widths=0.5,\
                                    labels=[labelExp],\
                                    showmeans=True,\
                                    notch=True,\
                                    boxprops=boxprops,\
                                    flierprops=flierprops,\
                                    whiskerprops=whiskerprops,\
                                    medianprops=medianprops,\
                                    meanprops=meanprops
                                )
                plt.setp(bpl['boxes'], color=c)
                plt.setp(bpl['whiskers'], color=c)
                plt.setp(bpl['caps'], color=c)
                plt.setp(bpl['medians'], color=c)

                sub_exp_box_plot_offset += sub_exp_box_plot_offset_step


            exp_box_plot_offset += exp_box_plot_offset_step

        # Legend
        plt.legend(fontsize=self.legendFontSize)
        plt.grid()
        # Show
        plt.show()




    def plotMccFunctionEinit(self):
        # Plotting a curve of the MCC as a function of epoch_init_dec_loss
        mcc_for_fixed_epoch_init_dec_loss = {}
        for importance_dec in self.vals_importance_dec:
            for epoch_init_dec_loss in self.vals_epoch_init_dec_loss:
                # Getting the results for that combination of hyperparams
                results_one_combination = self.results_combination_hyperparams[(importance_dec, epoch_init_dec_loss)]

                # Getting the metrics for that combination
                print("For combination (e_init, \u03B3) = ({}, {}) we have: ".format(importance_dec, epoch_init_dec_loss))
                mean_last_mcc, std_last_mcc = get_mean_metrics(results_one_combination['LastEpochMetrics_PerRep'], metric_type='MCC')


                # Plotting the curves
                if (importance_dec not in mcc_for_fixed_epoch_init_dec_loss):
                    mcc_for_fixed_epoch_init_dec_loss[importance_dec] = {
                                                                            'MCCMeans': [mean_last_mcc],
                                                                            'MCCStds': [std_last_mcc],
                                                                            'EpochInitDECLoss': [epoch_init_dec_loss]
                                                                        }
                else:
                    mcc_for_fixed_epoch_init_dec_loss[importance_dec]['MCCMeans'].append(mean_last_mcc)
                    mcc_for_fixed_epoch_init_dec_loss[importance_dec]['MCCStds'].append(std_last_mcc)
                    mcc_for_fixed_epoch_init_dec_loss[importance_dec]['EpochInitDECLoss'].append(epoch_init_dec_loss)

        # Doing the plot
        plt.figure()
        for importance_dec in mcc_for_fixed_epoch_init_dec_loss:
            x = mcc_for_fixed_epoch_init_dec_loss[importance_dec]['EpochInitDECLoss']
            y = mcc_for_fixed_epoch_init_dec_loss[importance_dec]['MCCMeans']
            yerr = mcc_for_fixed_epoch_init_dec_loss[importance_dec]['MCCStds']
            plt.errorbar(x=x, y=y, yerr=yerr, label="\u03B3 = {}".format(importance_dec))
        plt.title("MCC as a function of epoch_init_dec_loss")
        plt.xlabel("e\N{LATIN SUBSCRIPT SMALL LETTER I}\N{LATIN SUBSCRIPT SMALL LETTER N}\N{LATIN SUBSCRIPT SMALL LETTER I}\N{LATIN SUBSCRIPT SMALL LETTER T}")
        plt.ylabel("MCC")
        plt.xticks(fontsize=self.axisTickSize)
        plt.yticks(fontsize=self.axisTickSize)
        plt.legend(fontsize=self.legendFontSize)
        plt.show()


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
    ap.add_argument('--study_hyperparams', default="False", help="Boolean indicating if we want to start an experiment from scratch or use the results of a pre-computed experiment", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    study_hyperparams = args['study_hyperparams']
    if (study_hyperparams.lower() == 'true'):
        study_hyperparams = True
    else:
        study_hyperparams = False

    #==========================================================================#
    # Creating an instance of the experiment
    exp_2 = Experiment2(study_hyperparams)

    # Running the experiment
    exp_2.run()

    # Plotting the results matrix
    exp_2.plotResultsMatrix()

    # Plotting the MCC as a function of gamma (importance of DEC)
    exp_2.plotMccFunctionGamma()

    # Plotting box plots of MCC as a function of gamma (importance DEC)
    # Sub-exps colors
    sub_exps_colors = {
                            0: "xkcd:teal",
                            1: "xkcd:blue",
                            5: "xkcd:orange",
                            10: "xkcd:green",
                            20: "xkcd:red",
                            int(exp_2.base_parameters_exp['nb_epochs']/2.): "xkcd:purple",
                      }

    # Doing the box plot
    exp_2.boxPlotsByImportanceDEC(sub_exps_colors=sub_exps_colors)

    # Plotting the MCC as a function of e_init (epoch from which DEC is activated)
    exp_2.plotMccFunctionEinit()


if __name__=='__main__':
    main()
