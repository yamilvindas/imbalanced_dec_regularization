#!/usr/bin/env python3
"""
    Trains a single simple model using a synthetic dataset.
"""
import json
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import torch
from src.utils.DEC import DECLoss
from src.utils.tools import get_mean_metrics
from src.models.simple_model import SimpleClassifier
from torch.utils.data.sampler import SubsetRandomSampler
from src.utils.synthetic_data_generation import generate_synthetic_data, SyntheticDataset

class TrainSingleModel(object):
    def __init__(self, parameters_exp):
        """
            Class that trains a simple deep neural network with on a synthetic
            dataset. The model can be regularized with DEC.

            Arguments:
            ----------
            parameters_exp: dict
                Dictionary containing the parameters of the experiment:
                    * dataset_type: str
                        Type of synthetic dataset to use. Four options:
                            - balanced_separated
                            - imbalanced_separated
                            - balanced_mixed
                            - imbalanced_mixed
                    * use_dec: bool
                        True if DEC regularization has to be applied.
                    * nb_repetitions: int
                        Number of times to repeat the experiment.
                    * lr: float
                        Learning rate for the training
                    * batch_size: int
                        Number of samples per batch.
                    * nb_epochs: int
                        Number of complete passes to do over all the dataset
                    * importance_dec: float
                        Importance of DEC regularization (if used)
                    * epoch_init_dec_loss: int
                        Epoch from which DEC will be activated (if used)
        """
        #======================================================================#
        #==================== Definition of some attributes ====================
        #======================================================================#
        # Type of dataset to use
        self.dataset_type = parameters_exp['dataset_type']
        # Use DEC
        self.use_dec = parameters_exp['use_dec']
        # Number of repetitions
        self.nb_repetitions = parameters_exp['nb_repetitions']
        # Training parameters
        self.lr = parameters_exp['lr']
        self.batch_size = parameters_exp['batch_size']
        self.nb_epochs = parameters_exp['nb_epochs']
        self.criterion = torch.nn.CrossEntropyLoss()
        # Model parameters
        self.input_dim = 3
        self.enc_dim = 2
        self.nb_classes = 3
        # DEC hyper-parameters
        if (self.use_dec):
            self.importance_dec = parameters_exp['importance_dec']
            self.epoch_init_dec_loss = parameters_exp['epoch_init_dec_loss']
        # Device to use
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

    def generateSyntheticData(self):
        """
            Generates a synthetic 3D dataset based on the chosen dataset type
        """
        # Creating synthetic data points
        self.X_data, self.Y_data = generate_synthetic_data(self.dataset_type)

    def createDataLoader(self):
        """
            Creates a pytorch train and test loader using the synthetic data
        """
        # Splitting the data between train and test
        nb_total_samples = len(self.X_data)
        idxs_samples = list(range(nb_total_samples))
        random.shuffle(idxs_samples)
        # Traing samples
        X_data_train, Y_data_train = [], []
        train_data_percentage = 0.8
        nb_train_samples = int(nb_total_samples*train_data_percentage)
        for i in range(nb_train_samples):
            X_data_train.append(self.X_data[idxs_samples[i]])
            Y_data_train.append(self.Y_data[idxs_samples[i]])
        train_ds = SyntheticDataset(X_data_train, Y_data_train)
        # Testing samples
        X_data_test, Y_data_test = [], []
        for i in range(nb_train_samples, nb_total_samples):
            X_data_test.append(self.X_data[idxs_samples[i]])
            Y_data_test.append(self.Y_data[idxs_samples[i]])
        test_ds = SyntheticDataset(X_data_test, Y_data_test)

        # Creating the dataloaders
        # Training set
        train_indices = list(range(0, len(train_ds)))
        train_sampler = SubsetRandomSampler(train_indices)
        self.train_loader = torch.utils.data.DataLoader(
                                                            train_ds,\
                                                            batch_size=self.batch_size,\
                                                            sampler=train_sampler
                                                        )

        # Testing set
        test_indices = list(range(0, len(test_ds)))
        test_sampler = SubsetRandomSampler(test_indices)
        self.test_loader = torch.utils.data.DataLoader(
                                                            test_ds,\
                                                            batch_size=self.batch_size,\
                                                            sampler=test_sampler
                                                        )

    def modelCreation(self):
        """
            Creates a simple deep neural network model for classification
        """
        self.model = SimpleClassifier(self.input_dim, self.enc_dim, self.nb_classes).double().to(self.device)


    def createOptimizer(self, model_parameters):
        """
            Create a simple model for classification on a sythetic 3D dataset

            Arguments:
            ----------
            model_parameters: list of torch.nn.Parameters
                Parameters of the model to optimize

        """
        self.optimizer = torch.optim.Adam(model_parameters, lr=self.lr, weight_decay=1e-7)

    def initializeDecLoss(self):
        """
            Code strongly inspired from
            https://github.com/vlukiyanov/pt-dec/blob/master/ptdec/model.py
            lines 74-97
        """
        # Putting the right hidden dimension in the attributes of the DEC loss
        self.dec_loss = DECLoss(
                                cluster_number=self.nb_classes,
                                hidden_dimension=self.enc_dim, # Temporary, it will be updated later, during the initialization of DEC
                                alpha=1,
                           ).to(self.device)

        #======================================================================#
        # Computing the centroids using k-means
        kmeans = KMeans(n_clusters=self.nb_classes, n_init=20)
        self.model.train()
        features = []
        labels_list = []
        for batch in self.train_loader:
            sample_data, labels = batch
            sample_data, labels = sample_data.to(self.device), labels.to(self.device)
            features.append(self.model.encode(sample_data).detach().cpu())
            labels_list += list(labels.detach().cpu())
        final_features_embeddings = torch.cat(features).numpy()
        predicted = kmeans.fit_predict(final_features_embeddings)
        cluster_centers = torch.tensor(
                                        kmeans.cluster_centers_,
                                        dtype=torch.float,
                                        requires_grad=True
        ).to(self.device)

        with torch.no_grad():
            # initialise the cluster centers
            self.dec_loss.assignment.cluster_centers.copy_(cluster_centers)

        # Adding the cluster centers to the learnable parameters of the optimizer
        self.createOptimizer(list(self.model.parameters()) + [self.dec_loss.assignment.cluster_centers])

    def computeForwardPass(self, batch, current_epoch, keep_grad=True):
        """
            Does one single forward pass using the data in the given batch

            Arguments:
            ----------
            batch:
                Batch used to do the forward pass
            current_epoch: int
                Current epoch
            keep_grad: Boolean
                True if the user want to compute the gradient and use it to
                update the model parameters'
        """
        # Getting the data and the labels
        sample_data, labels = batch
        sample_data, labels = sample_data.to(self.device), labels.to(self.device)

        # Computing the DEC term
        if (self.use_dec):
            # It HAS TO BE COMPUTED BEFORE the forward pass
            if (keep_grad) and (current_epoch >= self.epoch_init_dec_loss):
                dec_loss = self.dec_loss(self.model.enc, sample_data)

        # Computing the classification loss
        out = self.model(sample_data) # Generate predictions
        classification_loss = self.criterion(out, labels.long()) # Calculate loss

        # Final loss
        loss = classification_loss
        if (self.use_dec):
            if (keep_grad) and (current_epoch >= self.epoch_init_dec_loss):
                loss += self.importance_dec*dec_loss

        # Getting the predictions
        y_true, y_pred = [], []
        for i in range(len(out)):
            true_class = int(labels[i])
            y_true.append(true_class)
            predicted_class = int(out[i].max(0)[1])
            y_pred.append(predicted_class)

        predictions = {
                        'TrueLabels': y_true,
                        'PredictedLabels': y_pred
                    }

        return loss, predictions


    def trainModel(self):
        """
            Trains a model (one single time) on the chosen synthetic dataset

            Arguments:
            ----------

        """
        # Creation of the dataloaders
        self.createDataLoader()

        # Creation of the model
        self.modelCreation()

        # Creation of the optimizer
        optimizer = self.createOptimizer(self.model.parameters())

        # Data structures for the losses and the predictions
        loss_values = {
                        'Train': [0 for _ in range(self.nb_epochs)],
                        'Test': [0 for _ in range(self.nb_epochs)]
                      }
        predictions_results = {}
        for dataset_split in ['Train', 'Test']:
            predictions_results[dataset_split] = {}
            for type_labels in ['TrueLabels', 'PredictedLabels']:
                predictions_results[dataset_split][type_labels] =  [[] for _ in range(self.nb_epochs)]

        # Iterating over the epochs
        encodings_by_epochs = {
                                    'Train': {epoch: {'Points': [], 'Encodings': [], 'TrueLabels': [], 'PredictedLabels': []} for epoch in range(self.nb_epochs)},
                                    'Test': {epoch: {'Points': [], 'Encodings': [], 'TrueLabels': [], 'PredictedLabels': []} for epoch in range(self.nb_epochs)}
                              }
        cluster_centers_by_epoch = {}
        for epoch in range(self.nb_epochs):
            if (self.use_dec):
                # Initialize DEC LOSS
                if (epoch == self.epoch_init_dec_loss):
                    # Initializing the DEC loss
                    self.initializeDecLoss()
                    # Getting the cluster centers
                    cluster_centers = torch.clone(self.dec_loss.assignment.cluster_centers).detach().cpu()
                    cluster_centers_by_epoch[-1] = cluster_centers # Initial cluster centers (initialize with our method)
                elif (epoch < self.epoch_init_dec_loss):
                    self.dec_loss = None
            else:
                self.dec_loss = None
                embeddings_used_to_get_centroids = None

            # Train the model
            self.model.train()
            tmp_train_losses = []
            for batch in self.train_loader:
                # Zero the parameters gradients
                self.optimizer.zero_grad()

                # Forward pass
                train_loss, train_predictions = self.computeForwardPass(batch, epoch, keep_grad=True)
                tmp_train_losses.append(train_loss.detach().data.cpu().numpy())

                # Backward pass for the gradient computation
                train_loss.backward()

                # Updating the weights
                self.optimizer.step()

                # Encoding the data
                with torch.no_grad():
                    sample_data, label = batch[0].to(self.device), batch[1].to(self.device)
                    encodings = self.model.encode(sample_data)
                    for i in range(len(sample_data)):
                        encodings_by_epochs['Train'][epoch]['Points'].append(sample_data[i].cpu().detach().numpy())
                        encodings_by_epochs['Train'][epoch]['Encodings'].append(encodings[i].cpu().detach().numpy())
                        encodings_by_epochs['Train'][epoch]['PredictedLabels'].append(train_predictions['PredictedLabels'][i])
                        encodings_by_epochs['Train'][epoch]['TrueLabels'].append(train_predictions['TrueLabels'][i])

                # Updating the predictions results of the current epoch
                predictions_results['Train']['TrueLabels'][epoch] += train_predictions['TrueLabels']
                predictions_results['Train']['PredictedLabels'][epoch] += train_predictions['PredictedLabels']
            loss_values['Train'][epoch] = np.mean(tmp_train_losses)

            # Getting the cluster centers
            if (self.use_dec):
                if (self.dec_loss is not None):
                    cluster_centers = torch.clone(self.dec_loss.assignment.cluster_centers).detach().cpu()
                    cluster_centers_by_epoch[epoch] = cluster_centers

            # Testing the model
            with (torch.no_grad()):
                self.model.eval()
                tmp_test_losses = []
                for batch in self.test_loader:
                    # Forward pass
                    test_loss, test_predictions = self.computeForwardPass(batch, epoch, keep_grad=False)
                    tmp_test_losses.append(test_loss.detach().data.cpu())

                     # Encoding the data
                    sample_data, label = batch[0].to(self.device), batch[1].to(self.device)
                    encodings = self.model.encode(sample_data)
                    for i in range(len(sample_data)):
                        encodings_by_epochs['Test'][epoch]['Points'].append(sample_data[i].cpu().detach().numpy())
                        encodings_by_epochs['Test'][epoch]['Encodings'].append(encodings[i].cpu().detach().numpy())
                        encodings_by_epochs['Test'][epoch]['PredictedLabels'].append(test_predictions['PredictedLabels'][i])
                        encodings_by_epochs['Test'][epoch]['TrueLabels'].append(test_predictions['TrueLabels'][i])

                    # Updating the predictions results of the current epoch
                    predictions_results['Test']['TrueLabels'][epoch] += test_predictions['TrueLabels']
                    predictions_results['Test']['PredictedLabels'][epoch] += test_predictions['PredictedLabels']

                loss_values['Test'][epoch] = np.mean(tmp_test_losses)
                print("================================================================================")
                print("METRICS\n")
                print("\n=======>Train loss at epoch {} is {}".format(epoch, loss_values['Train'][epoch]))
                print("\t\tTest loss at epoch {} is {}".format(epoch, loss_values['Test'][epoch]))
                print("\n=======>Train F1 Score at epoch {} is {}\n".format(epoch, f1_score(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch], average='micro')))
                print("\t\tTest F1 Score at epoch {} is {}".format(epoch, f1_score(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch], average='micro')))
                print("\n=======>Train MCC at epoch {} is {}\n".format(epoch, matthews_corrcoef(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch])))
                print("\t\tTest MCC at epoch {} is {}".format(epoch, matthews_corrcoef(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch])))
                print("\n=======>Train accuracy at epoch {} is {}\n".format(epoch, accuracy_score(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch])))
                print("\t\tTest accuracy at epoch {} is {}".format(epoch, accuracy_score(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch])))
                print("================================================================================\n\n")

        # Saving the last epoch metrics
        last_epoch_metrics = {'Train': {'MCC': 0, 'F1-Score': 0, 'Accuracy': 0}, 'Test': {'MCC': 0, 'F1-Score': 0, 'Accuracy': 0}}
        last_epoch_metrics['Train']['F1-Score'] = f1_score(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch], average='micro')
        last_epoch_metrics['Test']['F1-Score'] = f1_score(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch], average='micro')
        last_epoch_metrics['Train']['MCC'] = matthews_corrcoef(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch])
        last_epoch_metrics['Test']['MCC'] = matthews_corrcoef(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch])
        last_epoch_metrics['Train']['Accuracy'] = accuracy_score(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch])
        last_epoch_metrics['Test']['Accuracy'] = accuracy_score(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch])


        return loss_values, encodings_by_epochs, cluster_centers_by_epoch, last_epoch_metrics


    def repeatedHoldout(self):
        """
            Repeats the experiment self.nb_repetitions times
        """
        # Creating the synthetic data
        self.generateSyntheticData()
        # Training the model
        self.list_loss_values = []
        self.list_encodings_by_epochs = []
        self.list_cluster_centers_by_epoch = []
        self.list_last_epoch_metrics = []
        for rep in range(self.nb_repetitions):
            # Doing the training
            tmp_loss_values,\
            tmp_encodings_by_epochs,\
            tmp_cluster_centers_by_epoch,\
            tmp_last_epoch_metrics = self.trainModel()
            # Storing the values of the embeddings and curves
            self.list_loss_values.append(tmp_loss_values)
            self.list_encodings_by_epochs.append(tmp_encodings_by_epochs)
            self.list_cluster_centers_by_epoch.append(tmp_cluster_centers_by_epoch)
            self.list_last_epoch_metrics.append(tmp_last_epoch_metrics)


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
    default_parameters_file = "../../parameters_files/single_train_NB.json"
    ap.add_argument('--parameters_file', default=default_parameters_file, help="Parameters for the experiment", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    parameters_file = args['parameters_file']
    with open(parameters_file) as jf:
        parameters_exp = json.load(jf)

    #==========================================================================#
    # Creation of the experiment instance
    single_train_exp = TrainSingleModel(parameters_exp)

    # Training the model
    single_train_exp.repeatedHoldout()

    # Print the mean metrics
    get_mean_metrics(single_train_exp.list_last_epoch_metrics, metric_type='MCC')
    get_mean_metrics(single_train_exp.list_last_epoch_metrics, metric_type='F1-Score')
    get_mean_metrics(single_train_exp.list_last_epoch_metrics, metric_type='Accuracy')

if __name__=='__main__':
    main()
