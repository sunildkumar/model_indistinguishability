import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

from .model_inferface import ModelInterface


class ViTModel(ModelInterface):
    def __init__(
        self,
        model_name="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        feature_extractor_name="google/vit-base-patch16-224-in21k",
    ):
        self.model_name = model_name
        self.feature_extractor_name = feature_extractor_name
        self.model = None
        self.feature_extractor = None

    def load_model(self):
        self.model = ViTForImageClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to("cuda")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            self.feature_extractor_name,
            force_download=True,
            do_rescale=False,
        )

    def teardown_model(self):
        if self.model is not None:
            self.model.cpu()
            torch.cuda.empty_cache()

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not loaded")

        inputs = self.feature_extractor(images=images, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model(**inputs)
        predicted_classes = torch.argmax(outputs.logits, dim=1)
        return predicted_classes

    @property
    def batch_size(self):
        return 64

    def __str__(self):
        return f"{self.__class__.__name__}"
