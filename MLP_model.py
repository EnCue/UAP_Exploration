import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules import loss as LossFunction

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
        )


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


def train_MLP(model, trainset_loader: DataLoader, testset_loader: DataLoader, optimizer, loss_function: LossFunction, epochs=5):
    count = 0
    # accuracy_list = []

    # Lists for knowing classwise accuracy

    for epoch in range(epochs):
        for inputs, labels in trainset_loader:
            # Transfering images and labels to GPU if available
            inputs, labels = inputs.to('mps'), labels.to('mps')

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()
            
            # Forward pass 
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            #Propagating the error backward
            loss.backward()
            
            # Optimizing the parameters
            optimizer.step()

            count += 1
        
        # Testing the model
        
            if count % 500 == 0:
                total = 0
                correct = 0
            
                for inputs, labels in testset_loader:
                    inputs, labels = inputs.to('mps'), labels.to('mps')

                    outputs = model(inputs)
                
                    predictions = torch.max(outputs, 1)[1]
                    correct += (predictions == labels).sum()
                
                    total += len(labels)
                
                accuracy = correct * 100 / total
                # accuracy_list.append(accuracy)
            
            if count % 1000 == 0:
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))