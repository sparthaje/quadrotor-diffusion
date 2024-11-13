import numpy as np


class Normalizer:
    def __call__(self, array):
        raise NotImplementedError("Subclasses must implement the __call__ method.")

    def undo(self, normalized_array):
        raise NotImplementedError("Subclass must implement the unod method")


class NoNormalizer:
    """
    Trains without normalization on dataset
    """

    def __call__(self, array):
        return array

    def undo(self, normalized_array):
        return normalized_array


class GuassianNormalizer(Normalizer):
    def __init__(self, mean, variance):
        """
        Normalizes using z-score
        mean / variance: n item arrays for each dimension of the data
        """
        self.mean = mean
        self.variance = variance

    def __call__(self, array):
        normalized_array = (array - self.mean) / np.sqrt(self.variance)
        return normalized_array

    def undo(self, normalized_array):
        original_array = (normalized_array * np.sqrt(self.variance)) + self.mean
        return original_array
