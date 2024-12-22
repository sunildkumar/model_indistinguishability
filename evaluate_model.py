from collections import Counter

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CIFAR10Testset
from model_zoo.contrastive_model import ContrastiveLearningModel
from model_zoo.conv_next import ConvNextModel
from model_zoo.swin import SwinModel
from model_zoo.vit import ViTModel
from model_zoo.vit16_lora import ViT16LoraModel


def evaluate_model(model, batches_to_eval=None):
    model.load_model()

    transform = transforms.ToTensor()

    dataset = CIFAR10Testset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=model.batch_size, shuffle=False)

    predicted_labels = []
    true_labels = []

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating model")):
        if batches_to_eval is not None and i >= batches_to_eval:
            break
        images, labels, filenames = batch
        predicted_classes = model.predict(images)
        predicted_labels.extend(predicted_classes.tolist())
        true_labels.extend(labels.tolist())

    accuracy = sum(p == t for p, t in zip(predicted_labels, true_labels)) / len(
        true_labels
    )
    print(f"Model: {model} Accuracy: {accuracy}")

    # Print the distribution of the predicted labels
    label_distribution = Counter(predicted_labels)
    print("Predicted label distribution:", label_distribution)

    model.teardown_model()


if __name__ == "__main__":
    print("evaluating models")

    vit_model = ViTModel()
    # evaluate_model(vit_model)

    convnext_model = ConvNextModel()
    # evaluate_model(convnext_model)

    swin_model = SwinModel()
    # evaluate_model(swin_model)

    vit16_lora_model = ViT16LoraModel()
    # evaluate_model(vit16_lora_model)

    contrastive_model = ContrastiveLearningModel()
    evaluate_model(contrastive_model)
