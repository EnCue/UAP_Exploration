import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules import loss as LossFunction

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


def train_CNN(model, trainset_loader: DataLoader, testset_loader: DataLoader, optimizer, loss_function: LossFunction, epochs=5):
    count = 0
    # accuracy_list = []

    # Lists for knowing classwise accuracy

    for epoch in range(epochs):
        for inputs, labels in trainset_loader:
            
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
