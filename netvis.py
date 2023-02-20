import math
from random import randrange

import torch
from torch import nn

from dataset import valid_dataloader


class FullyConnectedNetwork(nn.Module):
    def __init__(self, image_shape: tuple[int, int, int] = (1, 28, 28), num_classes: int = 10):
        super().__init__()

        input_features = math.prod(image_shape)

        self.fc1 = nn.Linear(input_features, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, num_classes)
        self.relu3 = nn.ReLU()

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.logsoftmax(x)

        return x


def activation_hook(layer, input_, output):
    print(f'layer: {layer}, activations:\n{output}')


if __name__ == '__main__':
    model: FullyConnectedNetwork = torch.load('checkpoint.pth')
    print(f'architecture: {model}')

    # Register activation hooks.
    model.fc1.register_forward_hook(activation_hook)
    model.relu1.register_forward_hook(activation_hook)

    model.fc2.register_forward_hook(activation_hook)
    model.relu2.register_forward_hook(activation_hook)

    model.fc3.register_forward_hook(activation_hook)
    model.relu3.register_forward_hook(activation_hook)

    model.logsoftmax.register_forward_hook(activation_hook)

    # Inference.
    with torch.no_grad():
        batch, _ = next(iter(valid_dataloader))
        sample = batch[randrange(0, 255)]
        print(f'input: {sample}')
        print('layer activations:')
        prediction = model(sample.view(sample.shape[0], -1)).argmax(dim=1)
        print(f'prediction: {prediction}')
