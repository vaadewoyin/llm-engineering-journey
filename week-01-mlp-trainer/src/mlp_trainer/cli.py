# CLI app

import torch
import typer

from mlp_trainer.data_processing import load_and_preprocess_data
from mlp_trainer.model import MLP
from mlp_trainer.trainer import save_training_curve_plot, train_model

app = typer.Typer()

@app.command()
def train(
    batch_size: int = typer.Option(32,help = "Batch size for training"),
    epochs: int = typer.Option(2, help = "Number of epochs for training"),
    lr: float = typer.Option(1e-1, help = "Learning rate for optimizer"),
    hidden_dim: list[int] = typer.Option([2], help = "Hidden layer sizes"),
    optimizer: str = typer.Option("sgd", help = "Optimizer: sgd or adam"),
    lr_scheduler: bool = typer.Option(False, help = "Whether to use learning rate scheduler (ReduceLROnPlateau) "),
    seed: int = typer.Option(45, help = "Random seed for reproducibility")
    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = load_and_preprocess_data(batch_size, seed)
    model = MLP(input_dim = 54, hidden_dim = hidden_dim, output_dim = 7)
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer =="adam":
        opt = torch.optim.Adam(model.parameters(), lr = lr)
    else:
        raise typer.BadParameter(f"Unknown paramter: {optimizer}")

    train_losses, train_accs, val_losses, val_accs = train_model(train_loader, val_loader, test_loader,  device, epochs, model, opt, lr_scheduler)

    # Save learning curve to /output
    save_training_curve_plot(train_losses, train_accs, val_losses, val_accs)


if __name__ == "__main__":
    app()


