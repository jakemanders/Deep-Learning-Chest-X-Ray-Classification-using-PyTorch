from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_transforms(image_size=(64, 64)):
    return Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def make_dataloaders(
    data_dir: str,
    image_size=(64, 64),
    batch_size: int = 32,
    num_workers: int = 0,
    validation_split: float = 0.2,
):
    root = Path(data_dir)
    transform = make_transforms(image_size)

    if (root / 'train').is_dir() and (root / 'val').is_dir():
        train_dataset = ImageFolder(root / 'train', transform=transform)
        val_dataset = ImageFolder(root / 'val', transform=transform)
    else:
        dataset = ImageFolder(root, transform=transform, is_valid_file=lambda x: x.lower().endswith(('.png', '.jpg', '.jpeg')))
        val_size = int(validation_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device):
    model.train()
    losses = []
    correct = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = float(np.mean(losses)) if losses else 0.0
    accuracy = 100.0 * correct / len(dataloader.dataset)
    return avg_loss, accuracy


def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            losses.append(loss.item())
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = float(np.mean(losses)) if losses else 0.0
    accuracy = 100.0 * correct / len(dataloader.dataset)
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    n_epochs: int,
):
    train_losses, val_losses = [], []
    train_acc, val_acc = [], []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)

        print(
            f'Epoch {epoch}/{n_epochs}: train_loss={train_loss:.4f}, train_acc={train_accuracy:.2f}%, '
            f'val_loss={val_loss:.4f}, val_acc={val_accuracy:.2f}%'
        )

    return train_losses, val_losses, train_acc, val_acc


def train_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    n_epochs: int,
    patience: int = 5,
):
    best_val_accuracy = 0.0
    best_state = None
    counter = 0

    train_losses, val_losses = [], []
    train_acc, val_acc = [], []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)

        print(
            f'Epoch {epoch}/{n_epochs}: train_loss={train_loss:.4f}, train_acc={train_accuracy:.2f}%, '
            f'val_loss={val_loss:.4f}, val_acc={val_accuracy:.2f}%'
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f'No validation improvement for {patience} epochs. Stopping early.')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, train_acc, val_acc


def plot_metrics(train_losses, val_losses, train_acc, val_acc, output_dir: str = None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.title('Loss curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(output_dir) / 'loss_curve.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(train_acc, label='train accuracy')
    plt.plot(val_acc, label='val accuracy')
    plt.title('Accuracy curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / 'accuracy_curve.png')
    plt.show()
