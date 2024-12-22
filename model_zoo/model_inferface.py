from abc import ABC, abstractmethod

import torch


class ModelInterface(ABC):
    @abstractmethod
    def load_model(self):
        """
        Loads model from disk and puts it on GPU
        """
        pass

    def teardown_model(self):
        """
        Cleans up model from GPU
        """
        pass

    @abstractmethod
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predicts the class of the images and returns a tensor of shape (batch_size, ) of ints
        """
        pass

    @property
    @abstractmethod
    def batch_size(self):
        """
        Returns the maximum batch size we can use for inference on a GTX 4090 (24GB of VRAM)
        """
        pass

    @abstractmethod
    def __str__(self):
        pass
