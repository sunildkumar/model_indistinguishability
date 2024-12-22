import csv
import os
import pickle

from PIL import Image
from tqdm import tqdm

# Script to download and unpack CIFAR-10 test set into individual PNG files.
# Extracts images and creates a CSV mapping filenames to labels.


def download_cifar_test_batch():
    # download the dataset from the UT website
    download_link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    os.makedirs("./data", exist_ok=True)
    os.system(f"wget {download_link} -O ./data/cifar-10-python.tar.gz")
    os.system("tar -xvzf ./data/cifar-10-python.tar.gz -C ./data")


def clean_up():
    # removes all but the test batch
    os.system("rm -rf ./data/cifar-10-python.tar.gz")
    os.system("rm -rf ./data/cifar-10-batches-py")


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def extract_test_data(batch, output_dir):
    images = batch[b"data"]
    filenames = batch[b"filenames"]
    labels = batch[b"labels"]  # Extract labels from the batch

    img_size = 32
    num_images = len(images)

    # Reshape to (num_images, 32, 32, 3)
    images = images.reshape(num_images, 3, img_size, img_size).transpose(0, 2, 3, 1)

    os.makedirs(output_dir, exist_ok=True)

    # Open a CSV file to write the filename-label mapping
    with open(os.path.join(output_dir, "labels.csv"), mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["filename", "label"])  # Write header

        for i in tqdm(range(num_images), desc="Saving images"):
            img = Image.fromarray(images[i])
            filename = filenames[i].decode("utf-8")
            img.save(os.path.join(output_dir, filename))

            # Write the filename and label to the CSV
            csv_writer.writerow([filename, labels[i]])


def main():
    print("Downloading CIFAR-10 test set...")
    download_cifar_test_batch()

    print("Extracting CIFAR-10 test set...")
    # Path to the extracted CIFAR-10 folder
    test_batch_path = "./data/cifar-10-batches-py/test_batch"
    output_dir = "./data/cifar-10-test"

    # Process the test batch
    test_batch = unpickle(test_batch_path)
    extract_test_data(test_batch, output_dir)

    print("Cleaning up...")
    clean_up()

    print(f"Extracted {len(test_batch[b'filenames'])} images to {output_dir}")


if __name__ == "__main__":
    main()
