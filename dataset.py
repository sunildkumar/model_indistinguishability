import csv
import os

from PIL import Image
from torch.utils.data import Dataset


class CIFAR10Testset(Dataset):
    """
    CIFAR-10 test set dataset. Returns PIL image, label (int), filename (str).
    Example usage:
    dataset = CIFAR10Testset()
    """

    def __init__(self, root="./data/cifar-10-test", transform=None):
        self.root = root
        self.transform = transform
        self.data = []
        self.targets = []

        # Load the CSV file containing the filenames and labels
        csv_path = os.path.join(root, "labels.csv")
        with open(csv_path, mode="r") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header
            for row in csv_reader:
                filename, label = row
                self.data.append(filename)
                self.targets.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns PIL image, label (int), filename (str)
        """
        # Load image
        img_path = os.path.join(self.root, self.data[index])
        image = Image.open(img_path)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Return image and label
        return image, self.targets[index], self.data[index]
