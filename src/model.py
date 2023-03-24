import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2,2)
        
        # Linear layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512*14*14, 2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.dp1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dp2 = nn.Dropout(dropout)
        
        
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.relu1(self.pool1(self.bn1(self.conv1(x))))
        x = self.relu2(self.pool2(self.bn2(self.conv2(x))))
        x = self.relu3(self.pool3(self.bn3(self.conv3(x))))
        x = self.relu4(self.pool4(self.bn4(self.conv4(x))))
        
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.dp1(x)
        
        x = self.fc2(x)
        x = self.bn6(x)
        x = self.dp2(x)
        
        x = self.fc3(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
