import torch
import pandas as pd
import hydra
from omegaconf import DictConfig
from train import create_dataloader, train_model, visualize_training
from evaluate import evaluate_model
from data import load_and_preprocess_data
from model import SectorClassifier
from api import preprocess_new_company, predict_sector

@hydra.main(config_path='../../configs', config_name="config.yaml")
def main(cfg: DictConfig):
    print(f"Resolved raw data path: {cfg.data.raw_path}")
    seed = torch.manual_seed(cfg.misc.random_seed)
    print(f"Using seed: {seed} for reproducibility")
    print(f"Learning rate: {cfg.training.learning_rate}")
    
    # Load and preprocess the dataset
    file_path = cfg.data.raw_path
    column_transformer, X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(file_path)

    # Create DataLoaders
    batch_size = cfg.training.batch_size
    train_loader = create_dataloader(X_train, y_train, batch_size)
    val_loader = create_dataloader(X_val, y_val, batch_size)
    test_loader = create_dataloader(X_test, y_test, batch_size)

    # Initialize model, criterion, and optimizer
    input_size = X_train.shape[1]
    num_classes = y_train.shape[1]
    model = SectorClassifier(input_size, num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Train the model
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, cfg.training.epochs)
    visualize_training(train_losses, val_losses,cfg.training.training_image)

    # Evaluate the model on the test set
    print("Evaluating model...")
    evaluate_model(model, test_loader)

    # Save the model
    model_path = cfg.model.save_path
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Predict sector for a new company
    print("Predicting sector for a new company...")
    new_company_transformed = preprocess_new_company(cfg.api.new_company, column_transformer)
    sector_index = predict_sector(model, new_company_transformed)
    sector_name = pd.get_dummies(pd.read_csv(file_path)['Sector']).columns[sector_index]
    print(f"Predicted Sector: {sector_name}")

if __name__ == "__main__":
    main()
