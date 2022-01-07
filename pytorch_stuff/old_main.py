import torch
from torchvision import datasets  # pre-made datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.io as torchio
import cv2
import random
import os
import math

fashionMNIST_training = datasets.FashionMNIST(
    root="data/fashionMNIST",
    train=True,  # training or testing data
    download=True,
    transform=ToTensor()  # converts dataset images into tensors
)

fashionMNIST_testing = datasets.FashionMNIST(
    root="data/fashionMNIST",
    train=False,  # training or testing data
    download=True,
    transform=ToTensor()  # converts dataset images into tensors
)

"""
data in datasets is retrieved via indexing/__getitem__(). Returns a tuple of (data, label), where label is the index
of what the data's label in the dataset's classes attribute
"""


def show_img_dataset(data_set):
    """Iterates through a dataset of images and shows random images with their corresponding labels"""
    while True:
        data, label_i = data_set[random.randint(0, len(data_set))]
        label = data_set.classes[label_i]

        if isinstance(data, torch.Tensor):
            data = data.numpy()[0]
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        print(data, data.shape)
        data = cv2.resize(data, (300, 300))
        data = cv2.putText(data, label, (0, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
        cv2.imshow("img", data)
        cv2.waitKey(1000)


class FlowerDataset(Dataset):
    def __init__(self, root="data/flowers", train=True, download=False, transform=None, target_transform=None):
        self.root = root
        self.classes = os.listdir(root)
        self.reverse_labels = {label: index for index, label in enumerate(self.classes)}
        self.transform = transform
        self.target_transform = target_transform
        # number of each type of image (species of flower)
        self.image_numbers = {label: len(os.listdir(os.path.join(root, label))) for label in self.classes}

    def __len__(self):
        return sum(self.image_numbers.values())

    def __getitem__(self, index):
        for i, label in enumerate(self.classes):
            if index > self.image_numbers[label]:
                index -= self.image_numbers[label]
            else:
                print(index)
                path = os.path.join(self.root, label)
                path = os.path.join(path, os.listdir(path)[index])
                image = cv2.imread(path)  # would use torchvision.io.read_image() but it errors
                if self.transform:
                    image = self.transform(image)
                if self.target_transform:
                    image = self.target_transform(image)
                return image, i


flower_testing = FlowerDataset(
    train=False,  # training or testing data
    download=True,
    transform=ToTensor()  # converts dataset images into tensors
)

show_img_dataset(flower_testing)
