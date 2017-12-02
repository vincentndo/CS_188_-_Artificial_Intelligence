import numpy as np 
import data_classification_utils
from util import raiseNotDefined
import random

class Perceptron(object):
    def __init__(self, categories, numFeatures):
        """categories: list of strings 
           numFeatures: int"""
        self.categories = categories
        self.numFeatures = numFeatures

        """YOUR CODE HERE"""
        
        self.weight_vectors = np.ones((len(categories), numFeatures), "float")


    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: category with maximum score, must be from self.categories"""

        """YOUR CODE HERE"""

        index = max( range(len(self.categories)), key = lambda x: np.dot(self.weight_vectors[x], sample) )
        return self.categories[index]


    def train(self, samples, labels):
        """samples: np.array of shape (numSamples, numFeatures)
           labels: list of numSamples strings, all of which must exist in self.categories 
           performs the weight updating process for perceptrons by iterating over each sample once."""

        """YOUR CODE HERE"""

        for i in range(len(samples)):
          s = samples[i]
          guess = self.classify(s)

          if guess != labels[i]:
            j = self.categories.index(guess)
            self.weight_vectors[j] -= s
            j = self.categories.index(labels[i])
            self.weight_vectors[j] += s
