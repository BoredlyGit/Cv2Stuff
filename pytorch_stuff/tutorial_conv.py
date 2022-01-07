# Uses the last 4 videos (5-8) of https://www.youtube.com/playlist?list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh
# Read notes/nn_notes.md
# Contents are the same as matching ipynb file, I just don't wanna retrain the model every time I change stuff

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
from torch.nn import functional as nn_func
from torch import optim

REBUILD_DATA = False  # Don't want to rerun data preprocessing each time program runs

print("starting!")


class DogsAndCatsData:
    # I have no idea why the tutorial didn't use Dataset and DataLoader but whatever
    IMG_SIZE = (50, 50)  # resize images, raw input images are in diff sizes & ratios
    CATS = "data/CatsAndDogs/PetImages/Cat"
    DOGS = "data/CatsAndDogs/PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    cat_count = 0  # balancing
    dog_count = 0

    def build_training_data(self):
        for label in self.LABELS.keys():
            print(f"CREATING DATA FROM: {label}")
            for file in tqdm(os.listdir(label)):  # tqdm just wraps an iterable in a progress bar
                path = os.path.join(label, file)
                # read & convert to grayscale, color is added data that is not needed
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:  # bad images
                    continue
                img = cv2.resize(img, self.IMG_SIZE)
                # np.eye() used to create one-hot vectors so that cat=[1,0] & dog=[0,1]. eye(n) outputs a n*n array with
                # a diagonal across it such that array[n][n] = 0. (Ex: eye(3) -> [[1,0,0][0,1,0][0,0,1]])
                self.training_data.append((img, np.eye(2)[self.LABELS[label]]))

                if label == self.DOGS:
                    self.dog_count += 1
                elif label == self.CATS:
                    self.cat_count += 1
        print(f"{self.cat_count} cats, {self.dog_count} dogs")

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)  # Save training data for later use


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5))
        self.conv2 = nn.Conv2d(32, 64, (5, 5))
        self.conv3 = nn.Conv2d(64, 128, (5, 5))

        """
        Convolutional layers output differently from linear ones, outputting a tensor of n*n*out_channels.
        Thus, they need to be flattened. However, this varies based on input size, so just shove in a 50x50 of zeros,
        get the output shape, flatten, and use that.
        - In this case, the output size (from a 50*50 input) is 512 (2*2*128)

        - This reshapes into a 4d array because the input format is:
         [
            [
                [[data], 
                [data]],

                [[data_2],
                 [data_2]], 
            ], 
            [label, label_2]
        ]
        """
        self.conv_output_flattened_size = \
        self.run_conv_layers(torch.zeros(50, 50).view(-1, 1, 50, 50)).flatten().size()[0]
        self.zero_grad()

        self.fc1 = nn.Linear(self.conv_output_flattened_size, 512)
        self.fc2 = nn.Linear(512, 2)  # output layer

    def run_conv_layers(self, input_data):
        # (2, 2) is the max pool kernel shape
        features = nn_func.max_pool2d(nn_func.relu(self.conv1(input_data)), (2, 2))
        features = nn_func.max_pool2d(nn_func.relu(self.conv2(features)), (2, 2))
        features = nn_func.max_pool2d(nn_func.relu(self.conv3(features)), (2, 2))

        return features

    def forward(self, input_data):
        conv_features = self.run_conv_layers(input_data)
        # flatten into a tensor of tensors of flattened features
        conv_features = conv_features.view(-1, self.conv_output_flattened_size)
        activations = nn_func.relu(self.fc1(conv_features))
        output = self.fc2(activations)
        return nn_func.softmax(output, dim=1)


if REBUILD_DATA:
    DogsAndCatsData().build_training_data()

train_data = np.load("training_data.npy", allow_pickle=True)

images = Tensor(np.array([example[0] for example in train_data])).view(-1, 50, 50)
images = images / 255  # make grayscale values a percentage of 255 so that 0 < x < 1
labels = Tensor(np.array([example[1] for example in train_data]))

TEST_PERCENT = 0.10  # Use 10% of the data as tests

train_images = images[:-int((len(images) * TEST_PERCENT))]
train_labels = labels[:-int((len(images) * TEST_PERCENT))]
test_images = images[-int((len(images) * TEST_PERCENT)):]
test_labels = labels[-int((len(images) * TEST_PERCENT)):]

network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.001)

EPOCHS = 3
BATCH_SIZE = 100

# Training:
network.train()
for epoch in range(EPOCHS):
    # I have no idea why the tutorial didn't use Dataset and DataLoader but whatever
    total_loss = []
    for index in tqdm(range(0, len(train_data), BATCH_SIZE)):
        batch_images = train_images[index:index + BATCH_SIZE].view(-1, 1, 50, 50)
        batch_labels = train_labels[index:index + BATCH_SIZE]

        # optimizer.zero_grad() does the same thing IF optimizer was given all parameters (network.parameters())
        network.zero_grad()
        outputs = network(batch_images)
        loss = nn.MSELoss()(outputs, batch_labels)  # MSELoss is a callable class
        loss.backward()
        total_loss.append(loss.item())
        optimizer.step()
    print(f"epoch {epoch}, average loss: {sum(total_loss) / len(total_loss)}")

# print(Network().run_conv_layers(torch.zeros(50, 50).view(-1, 1, 50, 50)).flatten().shape)

# Testing
correct = 0
total = 0

with torch.no_grad():
    network.eval()
    for index in tqdm(range(len(test_images))):
        answer = torch.argmax(test_labels[index])  # argmax returns the INDEX of largest element
        prediction = torch.argmax(network(test_images[index].view(-1, 1, 50, 50)))
        #         print(test_labels[index], network(test_images[index].view(-1, 1, 50, 50)))
        if answer == prediction:
            correct += 1
        total += 1

print(f"Correct: {correct}, total: {total} | {(correct / total) * 100}%")
