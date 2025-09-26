"""medapp: A Flower / pytorch_msg_api app."""

import os

import torch

import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def load_data(data_path: str):
    """Load partition."""
    partition = load_from_disk(data_path)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def maybe_init_wandb(use_wandb: bool, wandbtoken: str) -> None:
    """Initialize Weights & Biases if specified in run_config."""
    if use_wandb:
        if not wandbtoken:
            print(
                "W&B token wasn't found. Set it by passing `--run-config=\"wandb-token='<YOUR-TOKEN>'\" to your `flwr run` command.",
            )
            use_wandb = False
        else:
            os.environ["WANDB_API_KEY"] = wandbtoken
            wandb.init(project="Flower-hackathon-MedApp")


def load_centralized_dataset(data_path: str):
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_from_disk(data_path)
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)
