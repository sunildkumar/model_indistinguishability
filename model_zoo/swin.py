import torch
from transformers import AutoImageProcessor, SwinForImageClassification

from .model_inferface import ModelInterface


class SwinModel(ModelInterface):
    def __init__(
        self,
        model_name="Weili/swin-base-patch4-window7-224-in22k-finetuned-cifar10",
        processor_name="microsoft/swin-base-patch4-window7-224",
    ):
        self.model_name = model_name
        self.processor_name = processor_name
        self.model = None
        self.processor = None

    def load_model(self):
        self.model = SwinForImageClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to("cuda")
        self.processor = AutoImageProcessor.from_pretrained(
            self.processor_name,
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

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model(**inputs)
        predicted_classes = torch.argmax(outputs.logits, dim=1)
        return predicted_classes

    @property
    def batch_size(self):
        return 96

    def __str__(self):
        return f"{self.__class__.__name__}"
