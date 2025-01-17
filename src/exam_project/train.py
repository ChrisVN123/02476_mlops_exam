# src/exam_project/train.py
import matplotlib.pyplot as plt
import torch
import wandb  # Import Weights & Biases
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(X, y, batch_size):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X.toarray(), dtype=torch.float32)  # Convert sparse to dense
    else:
        X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    log_to_wandb=False,
):
    train_losses = []
    val_losses = []

    if log_to_wandb:
        wandb.watch(model, log="all", log_freq=10)  # Track weights and gradients

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, torch.argmax(y_batch, dim=1))
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        # Print and log progress
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}"
        )

        if log_to_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_losses[-1],
                    "val_loss": val_losses[-1],
                }
            )

    return train_losses, val_losses


def visualize_training(train_losses, val_losses, path_result):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.savefig(f"{path_result}/training.png")

    # Log training visualization to W&B
    wandb.log({"training_curve": wandb.Image(f"{path_result}/training.png")})
