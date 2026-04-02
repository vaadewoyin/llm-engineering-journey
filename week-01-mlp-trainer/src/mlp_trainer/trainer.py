# Imports
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy


# Train one epoch
def train_one_epoch(model, data_loader, optimizer, accuracy_metric, device):
    model.train()
    epoch_loss = 0
    accuracy_metric.reset()
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # Forward
        logits = model(input)
        loss = F.cross_entropy(logits, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss = (epoch_loss) + loss / len(data_loader)
        preds = torch.argmax(logits, dim=1)
        accuracy_metric.update(preds, target)

    train_accuracy = accuracy_metric.compute()

    return epoch_loss.item(), train_accuracy.item()

# Evaluate model on validation/test set
def evaluate(model, data_loader, accuracy_metric, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for input, target in data_loader:
              input, target = input.to(device), target.to(device)
              logits = model(input)
              preds = torch.argmax(logits, dim=1)

              accuracy_metric.update(preds, target)
              loss = F.cross_entropy(logits, target)
              val_loss = val_loss + loss / len(data_loader)

        overall_accuracy = accuracy_metric.compute()
    return val_loss.item(), overall_accuracy.item()

# Model training
def train_model(train_loader, val_loader, test_loader, device, epochs,
                model, optimizer, scheduler = None):

    # Initialize accuracy metric
    train_accuracy = Accuracy(task='multiclass', num_classes=7).to(device)
    val_accuracy = Accuracy(task='multiclass', num_classes=7).to(device)
    test_accuracy = Accuracy(task='multiclass', num_classes=7).to(device)
    # Store metrics for visualization
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    # Instatiate learning rate scheduler
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
                f'Train Acc: {train_acc:.3f}% | Val Acc: {val_acc:.3f}%')

    # Check test accuracy
    _, test_acc = evaluate(model, test_loader, test_accuracy, device)
    print(f'\nTest Accuracy: {test_acc:.3f}%')

    return train_losses, train_accs, val_losses, val_accs

def save_training_curve_plot(train_losses, train_accs, val_losses, val_accs):
    # Plot Learning curves
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
