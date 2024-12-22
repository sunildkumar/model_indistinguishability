import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from .model_inferface import ModelInterface


class ViT16LoraModel(ModelInterface):
    def __init__(
        self,
        model_name="google/vit-base-patch16-224-in21k",
        adapter_name="yturkunov/cifar10_vit16_lora",
    ):
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.model = None
        self.feature_extractor = None

    def load_model(self):
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=10,  # CIFAR-10 has 10 classes
            id2label={
                i: label
                for i, label in enumerate(
                    [
                        "plane",
                        "car",
                        "bird",
                        "cat",
                        "deer",
                        "dog",
                        "frog",
                        "horse",
                        "ship",
                        "truck",
                    ]
                )
            },
            label2id={
                label: i
                for i, label in enumerate(
                    [
                        "plane",
                        "car",
                        "bird",
                        "cat",
                        "deer",
                        "dog",
                        "frog",
                        "horse",
                        "ship",
                        "truck",
                    ]
                )
            },
        )
        self.model.load_adapter(self.adapter_name)
        self.model.eval()
        self.model.to("cuda")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name, do_rescale=False
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
        return 128

    def __str__(self):
        return f"{self.__class__.__name__}"
