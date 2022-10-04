#!/usr/bin/env python3
"""
    This code defines the simple deep neural network used for classification
    (and unsupervised clustering) on the synthetic datasets
"""
import numpy as np
import torch.nn as nn

# For Xavier Normal initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        # nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, enc_dim):
        super(SimpleEncoder, self).__init__()
        # First linear layer (encoding)
        self.fc_enc = nn.Linear(in_features=input_dim, out_features=enc_dim)

        # Activation function
        self.softplus = nn.Softplus()

    def forward(self, input):
        # First linear layer (encoding)
        x = self.softplus(self.fc_enc(input))
        #print("Data shape after first pattern: ", x.shape)

        return x

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, enc_dim, nb_classes):
        super(SimpleClassifier, self).__init__()
        # First layer (encoding)
        self.enc = SimpleEncoder(input_dim, enc_dim)

        # Second linear layer (classification)
        self.fc_classif = nn.Linear(in_features=enc_dim, out_features=nb_classes)

        # Activation function
        self.softmax = nn.Softmax(dim=0)

    def encode(self, input):
        # First layer (encoding)
        x = self.enc(input)

        return x

    def forward(self, input):
        # First linear layer (encoding)
        x = self.enc(input)
        #print("Data shape after first pattern: ", x.shape)

        # Second linear layer (classification)
        x = self.softmax(self.fc_classif(x))
        #print("Data shape after second pattern: ", x.shape)

        return x
