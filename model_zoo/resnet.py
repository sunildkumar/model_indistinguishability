import torch
from torchvision import transforms

from .model_inferface import ModelInterface


class ResNet20Model(ModelInterface):
    def __init__(self):
        self.model = None

    def load_model(self):
        # Load the model from torchvision models
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
        )

        # Add more model loading logic as needed
        self.model.eval()
        self.model.to("cuda")

    def teardown_model(self):
        if self.model is not None:
            self.model.cpu()
            torch.cuda.empty_cache()

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not loaded")

        # Define the normalization transform
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Apply normalization
        images = normalize(images)
        images = images.to("cuda")

        with torch.no_grad():
            outputs = self.model(images)
        predicted_classes = torch.argmax(outputs, dim=1)
        return predicted_classes

    @property
    def batch_size(self):
        return 128  # Adjust based on model and hardware

    def __str__(self):
        return f"{self.__class__.__name__}"
