import torch
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification

from .model_inferface import ModelInterface


class ConvNextModel(ModelInterface):
    def __init__(
        self,
        model_name="ahsanjavid/convnext-tiny-finetuned-cifar10",
        feature_extractor_name="facebook/convnext-tiny-224",
    ):
        self.model_name = model_name
        self.feature_extractor_name = feature_extractor_name
        self.model = None
        self.feature_extractor = None

    def load_model(self):
        self.model = ConvNextForImageClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to("cuda")
        self.feature_extractor = ConvNextFeatureExtractor.from_pretrained(
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
        return 200

    def __str__(self):
        return f"{self.__class__.__name__}"
