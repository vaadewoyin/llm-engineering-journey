# Imports
import datetime
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics import Metric

# Reproducibilty
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # For CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(45)

# Train one epoch
def train_one_epoch(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer,
                    accuracy_metric: Metric, device: torch.device) -> tuple[float, float]:
    """ Trains the model for one epoch on the given data loader and returns loss and accuracy """
    model.train()
    epoch_loss = 0
    accuracy_metric.reset()
    for items in data_loader:
        input = items['input_ids']
        target = items['label']
        mask = items['attention_mask'].bool()
        input, target = input.to(device), target.to(device)

        # Forward
        logits = model(input, key_padding_mask=mask)
        loss = F.cross_entropy(logits, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss = (epoch_loss) + loss.item() / len(data_loader)
        preds = torch.argmax(logits, dim=1)
        accuracy_metric.update(preds, target)

    train_accuracy = accuracy_metric.compute()

    return epoch_loss, train_accuracy.item()

# Evaluate model on validation/test set
def evaluate(model: nn.Module, data_loader: DataLoader,
             accuracy_metric: Metric, device: torch.device) -> tuple[float, float]:
    """ Evaluates the model on the given data loader and returns loss and accuracy """
    model.eval()
    accuracy_metric.reset()
    with torch.no_grad():
        val_loss = 0
        for items in data_loader:
            input = items['input_ids']
            target = items['label']
            mask = items['attention_mask'].bool()
            input, target = input.to(device), target.to(device)
            logits = model(input, key_padding_mask=mask)
            preds = torch.argmax(logits, dim=1)

            accuracy_metric.update(preds, target)
            loss = F.cross_entropy(logits, target)
            val_loss = val_loss + loss.item() / len(data_loader)

        overall_accuracy = accuracy_metric.compute()
    return val_loss, overall_accuracy.item()

# Model training
def train_model(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                device: torch.device, epochs: int,
                model: nn.Module, optimizer: Optimizer, use_scheduler: bool = True):
    """
    Trains and Evaluate model for specified number of epochs and returns training and validation losses and accuracies for each epoch.
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (Dataloader): Test data loader
        device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda')
        epochs (int): Number of epochs to train the model
        model (nn.Module): Model to be trained
        optimizer (Optimizer): Optimizer for training the model
        use_scheduler (bool, optional): Enable scheduler (ReduceLR on plateau) Defaults to True.

    Returns:
        Train and validation losses and accuracies for each epoch.
    """

    # Initialize accuracy metric
    train_accuracy = Accuracy(task='multiclass', num_classes=7).to(device)
    val_accuracy = Accuracy(task='multiclass', num_classes=7).to(device)
    test_accuracy = Accuracy(task='multiclass', num_classes=7).to(device)
    # Store metrics for visualization
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, train_accuracy, device)
        val_loss, val_acc = evaluate(model, val_loader, val_accuracy, device)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if (epoch % 5 == 0) or (epoch == epochs -1):
            print(f'Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                f'Train Acc: {train_acc * 100:.3f}% | Val Acc: {val_acc * 100:.3f}%')

    # Check test accuracy
    _, test_acc = evaluate(model, test_loader, test_accuracy, device)
    print(f'\nTest Accuracy: {test_acc* 100:.3f}%')

    return train_losses, train_accs, val_losses, val_accs

def save_training_curve_plot(train_losses, train_accs, val_losses, val_accs):
    """ Saves the training curve plot for losses and accuracies. """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot losses and accuracies on subplots
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(val_accs, label='Validation Accuracy')

    for ax, title in zip(axes, ['Loss', 'Accuracy']):
        ax.set_xlabel('Epoch')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save plot
    base_dir = Path(__file__).resolve().parents[2]
    output_dir =base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = output_dir/ f"{time_stamp}"
    run_dir.mkdir(exist_ok=True)

    plt.savefig(run_dir/ "learning_curve.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {run_dir}")
    plt.close()

