import numpy as np


class Normalizer:
    def __call__(self, array):
        raise NotImplementedError("Subclasses must implement the __call__ method.")

    def undo(self, normalized_array):
        raise NotImplementedError("Subclass must implement the unod method")

    def __str__(self):
        raise NotImplementedError("__str__ not done")


class NoNormalizer:
    """
    Trains without normalization on dataset
    """

    def __call__(self, array):
        return array

    def undo(self, normalized_array):
        return normalized_array

    def __str__(self):
        return "NoNormalizer"


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

    def __str__(self):
        return f"GuassianNormalizer: µ = {self.mean}, s = {self.variance}"


class LinearNormalizer(Normalizer):
    def __init__(self, scalers: np.array, biases: np.array):
        self.scalers = scalers
        self.biases = biases

    def __call__(self, array: np.array) -> np.array:
        return array * self.scalers + self.biases

    def undo(self, scaled_array: np.array) -> np.array:
        return (scaled_array - self.biases) / self.scalers

    def __str__(self):
        return f"ScalerNormalizer: scalers = {self.scalers}, biases = {self.biases}"


class MinMaxNormalizer(Normalizer):
    def __init__(self, mins: np.array, maxes: np.array):
        """
        Normalizes between min and max so all data is [0, 1]
        NOTE: no clipping, so if a value exceeds min/max it can have value > 1
        mins / maxes: n item arrays for each dimension of the data
        """
        self.mins = mins
        self.maxes = maxes

    def __call__(self, array: np.array) -> np.array:
        normalized_array = (array - self.mins) / (self.maxes - self.mins)
        return normalized_array

    def undo(self, normalized_array: np.array) -> np.array:
        original_array = (self.maxes - self.mins) * normalized_array + self.mins
        return original_array

    def __str__(self):
        return f"MinMaxNormalizer: min={self.mins}\tmaxes={self.maxes}"


class NormalizerTuple(Normalizer):
    def __init__(self, normalizer_a: Normalizer, normalizer_b: Normalizer):
        """
        Used in datasets where theres two series of data to normalize
        """
        self.normalizer_a = normalizer_a
        self.normalizer_b = normalizer_b

    def __call__(self, array_a: np.array, array_b: np.array) -> tuple[np.array, np.array]:
        return self.normalizer_a(array_a), self.normalizer_b(array_b)

    def undo(self, normalized_array_a: np.array, normalized_array_b: np.array) -> tuple[np.array, np.array]:
        return self.normalizer_a.undo(normalized_array_a), self.normalizer_b.undo(normalized_array_b)

    def __str__(self):
        return f"Normalizer a: {self.normalizer_a}, Normalizer b: {self.normalizer_b}"
