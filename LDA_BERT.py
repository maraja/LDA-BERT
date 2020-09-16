from .model import *
from .utils import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import math

import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse


class LDA_BERT():
    """
    LDA_BERT class to provide vectorized sentences with LDA.
    This is a supervised learning approach and requires the dataset to be
    handed over before initializing.
    """

    def __init__(self, sentences, topics, token_lists, pct_data=1):
        self.sentences = sentences
        self.topics = topics
        self.pct_data = pct_data
        self.sentences = self.sentences[:math.floor(len(self.sentences)*pct_data)]
        self.token_lists = token_lists
        self.model = None

    def _compile(self):
        # Define the topic model object
        self.model = Topic_Model(k = self.topics, method = 'LDA_BERT')
        # # Fit the topic model by chosen method
        # self.model.fit(self.sentences, self.token_lists)
    
    def vectorize(self):
        if not self.model:
            self._compile()
        vectors = self.model.vectorize(self.sentences, self.token_lists)
        return vectors

# class Autoencoder:
#     """
#     Autoencoder for learning latent space representation
#     architecture simplified for only one hidden layer
#     """

#     def __init__(self, latent_dim=32, activation='relu', epochs=200, batch_size=128):
#         self.latent_dim = latent_dim
#         self.activation = activation
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.autoencoder = None
#         self.encoder = None
#         self.decoder = None
#         self.his = None

#     def _compile(self, input_dim):
#         """
#         compile the computational graph
#         """
#         input_vec = Input(shape=(input_dim,))
#         encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
#         decoded = Dense(input_dim, activation=self.activation)(encoded)
#         self.autoencoder = Model(input_vec, decoded)
#         self.encoder = Model(input_vec, encoded)
#         encoded_input = Input(shape=(self.latent_dim,))
#         decoder_layer = self.autoencoder.layers[-1]
#         self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
#         self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

#     def fit(self, X):
#         if not self.autoencoder:
#             self._compile(X.shape[1])
#         X_train, X_test = train_test_split(X)
#         self.his = self.autoencoder.fit(X_train, X_train,
#                                         epochs=200,
#                                         batch_size=128,
#                                         shuffle=True,
#                                         validation_data=(X_test, X_test), verbose=0)