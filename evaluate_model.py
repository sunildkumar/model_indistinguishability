import random

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CIFAR10Testset
from model_zoo.conv_next import ConvNextModel
from model_zoo.resnet import ResNet20Model
from model_zoo.swin import SwinModel
from model_zoo.vit import ViTModel
from model_zoo.vit16_lora import ViT16LoraModel


def evaluate_model(model):
    model.load_model()

    transform = transforms.ToTensor()

    dataset = CIFAR10Testset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=model.batch_size, shuffle=False)

    predicted_labels = []
    true_labels = []
    fnames = []

    for batch in tqdm(dataloader, desc="Evaluating model"):
        images, labels, filenames = batch
        predicted_classes = model.predict(images)
        predicted_labels.extend(predicted_classes.tolist())
        true_labels.extend(labels.tolist())
        fnames.extend(filenames)
    accuracy = sum(p == t for p, t in zip(predicted_labels, true_labels)) / len(
        true_labels
    )
    print(f"Model: {model} Accuracy: {accuracy}")

    model.teardown_model()

    return list(zip(fnames, predicted_labels, true_labels))


if __name__ == "__main__":
    print("evaluating models")
    random.seed(42)

    vit_model = ViTModel()
    vit_results = evaluate_model(vit_model)

    convnext_model = ConvNextModel()
    convnext_results = evaluate_model(convnext_model)

    swin_model = SwinModel()
    swin_results = evaluate_model(swin_model)

    vit16_lora_model = ViT16LoraModel()
    vit16_lora_results = evaluate_model(vit16_lora_model)

    resnet_model = ResNet20Model()
    resnet_results = evaluate_model(resnet_model)

    df = pd.DataFrame(
        columns=[
            "filename",
            "True label",
            "ViT Label",
            "ConvNext Label",
            "Swin Label",
            "ViT16Lora Label",
            "ResNet Label",
        ]
    )

    # populate the fnames and true labels
    for fname, predicted_label, true_label in vit_results:
        df.loc[fname, "filename"] = fname
        df.loc[fname, "True label"] = true_label

    # populate the vit labels
    for fname, predicted_label, true_label in vit_results:
        df.loc[fname, "ViT Label"] = predicted_label

    # populate the convnext labels
    for fname, predicted_label, true_label in convnext_results:
        df.loc[fname, "ConvNext Label"] = predicted_label

    # populate the swin labels
    for fname, predicted_label, true_label in swin_results:
        df.loc[fname, "Swin Label"] = predicted_label

    # populate the vit16 lora labels
    for fname, predicted_label, true_label in vit16_lora_results:
        df.loc[fname, "ViT16Lora Label"] = predicted_label

    # populate the resnet labels
    for fname, predicted_label, true_label in resnet_results:
        df.loc[fname, "ResNet Label"] = predicted_label

    df.to_csv("model_eval_results.csv", index=False)
