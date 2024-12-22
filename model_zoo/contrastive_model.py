import tensorflow as tf
import torch
from huggingface_hub import snapshot_download
from torchvision import transforms

from .model_inferface import ModelInterface


class ContrastiveLearningModel(ModelInterface):
    def __init__(self, model_repo="keras-io/supervised-contrastive-learning-cifar10"):
        self.model_repo = model_repo
        self.model_path = None
        self.model = None
        self.feature_extractor = None

    def load_model(self):
        # Download the model from Hugging Face Hub
        model_dir = snapshot_download(repo_id=self.model_repo)

        self.model = tf.keras.models.load_model(
            model_dir,
            compile=False,  # Don't load optimizer state
        )

    def teardown_model(self):
        if self.model is not None:
            del self.model
            tf.keras.backend.clear_session()

    def predict(self, images: tf.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not loaded")
        # images are torch.Size([batch_size, 3, 32, 32])

        # Scale images from [0, 1] to [0, 255]
        images = images * 255.0

        # Normalize the images
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        images = transform(images)

        # Apply the transformation to each image in the batch
        images = tf.cast(images, tf.float32)
        images = tf.transpose(images, (0, 2, 3, 1))
        predictions = self.model(images, training=False)
        predictions = tf.argmax(predictions, axis=1).numpy()
        return torch.from_numpy(predictions)

    @property
    def batch_size(self):
        return 128

    def __str__(self):
        return f"{self.__class__.__name__}"
