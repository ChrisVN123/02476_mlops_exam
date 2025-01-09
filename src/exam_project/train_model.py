import torch
import pandas as pd
import hydra
from omegaconf import DictConfig
from train import create_dataloader, train_model, visualize_training
from evaluate import evaluate_model
from data import load_and_preprocess_data
from model import SectorClassifier
from api import preprocess_new_company, predict_sector
from loguru import logger


# Configure the logger
logger.add("/Users/christianvanellnielsen/Library/CloudStorage/OneDrive-Personligt/Documents/Christian/DTU/5. Semester/02476_mlops_exam/results/app.log", level="DEBUG", rotation="10 MB")

@hydra.main(config_path='../../configs', config_name="config.yaml")
def main(cfg: DictConfig):
    try:
        logger.info(f"Resolved raw data path: {cfg.data.raw_path}")
        
        # Set seed for reproducibility
        seed = torch.manual_seed(cfg.misc.random_seed)
        logger.debug(f"Using seed: {cfg.misc.random_seed} for reproducibility")
        
        logger.info(f"Learning rate: {cfg.training.learning_rate}")
        
        # Load and preprocess the dataset
        file_path = cfg.data.raw_path
        logger.info("Loading and preprocessing data...")
        column_transformer, X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(file_path)
        logger.success("Data loaded and preprocessed successfully.")
        
        # Create DataLoaders
        batch_size = cfg.training.batch_size
        logger.info(f"Creating DataLoaders with batch size {batch_size}...")
        train_loader = create_dataloader(X_train, y_train, batch_size)
        val_loader = create_dataloader(X_val, y_val, batch_size)
        test_loader = create_dataloader(X_test, y_test, batch_size)
        
        # Initialize model, criterion, and optimizer
        input_size = X_train.shape[1]
        num_classes = y_train.shape[1]
        model = SectorClassifier(input_size, num_classes)
        logger.debug("Model initialized successfully.")
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        
        # Train the model
        logger.info("Starting training...")
        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, cfg.training.epochs)
        visualize_training(train_losses, val_losses, cfg.training.training_image)
        logger.success("Training completed.")
        
        # Evaluate the model
        logger.info("Evaluating model...")
        evaluate_model(model, test_loader)
        logger.success("Model evaluation completed.")
        
        # Save the model
        model_path = cfg.model.save_path
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Predict sector for a new company
        logger.info("Predicting sector for a new company...")
        new_company_transformed = preprocess_new_company(cfg.api.new_company, column_transformer)
        sector_index = predict_sector(model, new_company_transformed)
        sector_name = pd.get_dummies(pd.read_csv(file_path)['Sector']).columns[sector_index]
        logger.success(f"Predicted Sector: {sector_name}")
    
    except Exception as e:
        logger.exception("An error occurred during execution.")


if __name__ == "__main__":
    main()
