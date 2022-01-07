# Uses the first 4 videos of https://www.youtube.com/playlist?list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh
# Read notes/nn_notes.md

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
import torch.nn.functional as nn_functional
from torch import optim


train_data = datasets.MNIST("data/MNIST", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST("data/MNIST", train=False, download=True, transform=transforms.ToTensor())

"""
- DataLoaders
    - wrap Datasets in an iterable, yielding the data from the dataset in batches of batch_size. (for Mini-Batch 
      Gradient Descent)
      - Common batch_sizes range from 8-64
    - This data will be a list with 2 tensors, the first containing the image/data tensors and the second containing the 
      labels (in order), and can be accessed by iterating over the DataLoader
      - Because of this, the data input to the network is a tensor of tensors, even if there's only 1 tensor in it.
    - Shuffle should almost always be True
    
- Data Balancing - If a large part of your data is made up of the same label, the network will train itself to 
  "recognize" only that label, and not anything else, and get stuck trying to decrease cost, as attempting to 
  accommodate for anything that's not matched with that label results in higher cost.
"""
train_set = DataLoader(train_data, batch_size=10, shuffle=True)
test_set = DataLoader(test_data, batch_size=10, shuffle=True)


class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        #  Defining network layers
        """
        fc = Fully Connected (see nn.Linear)
        nn.Linear = "regular" neural network layer, flat layer with each neuron connected to all in next and previous 
                    layers (fully connected).
        in_features = number of neurons in previous layer (the inputs to this layer)
        out_features = number of neurons in this layer (the outputs of this layer)
        """
        # 784 inputs because input images are 28x28 (784 pixels = 784 input neurons)
        self.fc1 = nn.Linear(in_features=784, out_features=64)  # hidden layer 1
        self.fc2 = nn.Linear(in_features=64, out_features=64)   # hidden layer 2
        self.fc3 = nn.Linear(in_features=64, out_features=64)   # hidden layer 3
        self.fc4 = nn.Linear(in_features=64, out_features=10)   # output layer (10 possible outputs (0-9))

    def forward(self, input_data):
        """
        Actual operation (not learning) of the network, where data is passed through the layers
        - Called in the networks __call__() function.
        - Logic (if/else, for, etc.) can be implemented here just like a normal python function
          (ex: if {condition}, skip layer _). Can be used for task-specific layers.

        - Each layer is fed the reLU-d activations of the previous one
            - Using functional relu instead of nn's object cause it makes more sense as a standalone function
        """
        activations = self.fc1(input_data)  # nn.Linear and other layer classes are operated by calling them
        activations = self.fc2(nn_functional.relu(activations))
        activations = self.fc3(nn_functional.relu(activations))
        output_data = self.fc4(nn_functional.relu(activations))
        """
        - Softmax creates a probability distribution which sums to one (normalizes the output)
        - log_softmax takes the log of the softmax to create log-probabilities which are faster, more accurate 
          (numerically stable), & simpler
        - dim=1 because output data is a tensor containing tensors which contain the outputs.
            - the output tensors are in the order of the data that was fed in (input_data[i] -> output_data[i])  
        """
        output_data = nn_functional.log_softmax(output_data, dim=1)
        return output_data


network = MyNetwork()

# TRAINING
print("TRAINING")
"""
Optimizer to calculate gradient descent and change weights and biases (this does the "learning"). 
- network.parameters() returns the network's changeable attributes (weights and balances)
- lr = learning rate, size of gradient descent step, 
    - Too large learning rates can cause the network to not fully reach the "bottom" of a minimum (it "steps over" it)
    - Too small learning rates can cause the network to fall into a bad minimum, not being able to step "over" or out
      of it.
    - Decaying learning rate - Compromise between large and small rates, rate gets smaller over time.
"""
optimizer = optim.Adam(network.parameters(), lr=0.001)

EPOCHS = 3  # 1 epoch = 1 pass through entire dataset, diminishing return the more you have

for epoch in range(EPOCHS):
    total_loss = []
    for batch in train_set:
        data, labels = batch  # See above notes on DataLoaders if confused
        # Resets the network's gradient between batches - zeros the neurons' grad properties (see notes on
        # loss.backward()) (as params have already been updated based on the previous
        # gradient, and we don't want to add to it - instead we create a new one)
        network.zero_grad()

        """
        Feed data through network
        - view() is like reshape, but returns a reshaped "view" of the same memory instead of using more to create a 
          new Tensor
        - The network expects each piece of data to be flattened (1-dimensional), and to be given this data in a tensor
          containing the data tensors (hence 2 dimensions).
        - -1 indicates n (unknown) items in this dimension (batches can be different sizes).
        - Output is a tensor of tensors that contain the actual outputs for the data fed in, in the order it was fed        
        """
        output = network(data.view(-1, 784))
        # loss is based on 1 data point (batch here), cost is based on a sum/average of losses
        loss = nn_functional.nll_loss(output, labels)  # nll = negative log likelihood (network returns log softmax)
        """
        Connection between loss.backward(), network, and optimizer
        - The loss is implicitly connected to the network via its grad_fn attribute (all tensors have this attr). It 
          stores the computational history of the tensor, linking it back to all tensors (and operations) that were
          used to compute it (the "graph").
          - This is part of autograd (https://pytorch.org/docs/stable/notes/autograd.html,
                                      https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
        - When loss.backward() is called, it uses this to get to the neurons of the network and do backpropagation. The
          desired change/gradient of each neuron is stored in its grad attribute (an attr of all tensors, neurons are 
          probably tensors).
        - The optimizer then uses the values stored in each neuron's grad attribute to adjust the network's parameters.
        """
        loss.backward()  # gradient descent and backpropagation (calculate loss of each neuron)
        optimizer.step()  # parameter adjustment
        total_loss.append(loss.item())
    avg_loss = sum(total_loss)/len(total_loss)
    print(f"Epoch: {epoch}, loss: {avg_loss}")


# TESTING
print("TESTING")
correct = 0
total = 0

with torch.no_grad():  # do not keep track of the graph and gradient (b/c testing)
    network.eval()  # puts network in evaluation mode (opposite of network.train())

    for batch in test_set:
        data, labels = batch  # See above notes on DataLoaders if confused
        outputs = network(data.view(-1, 784))

        for i, output in enumerate(outputs):
            if torch.argmax(output) == labels[i]:
                correct += 1
            total += 1

print(f"Accuracy: {(correct/total)*100} ({correct}/{total})")
