import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification

from dataset import CIFAR10Testset

if __name__ == "__main__":
    # Load the model
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
    )
    model.to("cuda")

    # Load the feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
        force_download=True,
        do_rescale=False,
    )

    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),
        ]
    )

    dataset = CIFAR10Testset(transform=transform)

    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predicted_labels = []
    true_labels = []

    for batch in tqdm(dataloader, desc="Evaluating model"):
        images, labels, filenames = batch

        inputs = feature_extractor(
            images=images,
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = model(**inputs)
        predicted_classes = torch.argmax(outputs.logits, dim=1)

        predicted_labels.extend(predicted_classes.tolist())
        true_labels.extend(labels.tolist())

    accuracy = sum(p == t for p, t in zip(predicted_labels, true_labels)) / len(
        true_labels
    )
    print(f"Accuracy: {accuracy}")
