import torch
import torchvision

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5],
            std=[0.5],
        ),
    ]
)

train_dataset = torchvision.datasets.MNIST(root='dataset', download=True, train=True, transform=transform)
valid_dataset = torchvision.datasets.MNIST(root='dataset', download=True, train=False, transform=transform)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=4)
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=256, shuffle=True, num_workers=4)
