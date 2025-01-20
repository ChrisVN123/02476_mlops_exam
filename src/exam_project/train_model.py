import pickle

import hydra
import pandas as pd
import torch
import wandb  # Import Weights & Biases
from api import predict_sector, preprocess_new_company
from data import load_and_preprocess_data
from evaluate import evaluate_model
from loguru import logger
from model import SectorClassifier
from omegaconf import DictConfig
from train import create_dataloader, train_model, visualize_training

# Configure the logger
logger.add("results/app.log", level="DEBUG", rotation="10 MB")


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig):
    try:
        # Initialize W&B
        run = wandb.init(
            entity="dtumlops_24",
            project="sector-classification",
            config={
                "learning_rate": cfg.training.learning_rate,
                "batch_size": cfg.training.batch_size,
                "epochs": cfg.training.epochs,
                "random_seed": cfg.misc.random_seed,
            },
        )
        logger.info(f"Resolved raw data path: {cfg.data.raw_path}")

        # Set seed for reproducibility
        seed = torch.manual_seed(cfg.misc.random_seed)
        logger.debug(f"Using seed: {seed} for reproducibility")

        logger.info(f"Learning rate: {cfg.training.learning_rate}")

        # Load and preprocess the dataset
        file_path = cfg.data.raw_path
        logger.info("Loading and preprocessing data...")
        column_transformer, X_train, X_val, X_test, y_train, y_val, y_test = (
            load_and_preprocess_data(file_path)
        )
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
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            cfg.training.epochs,
            log_to_wandb=True,
        )
        visualize_training(train_losses, val_losses, cfg.training.training_image)
        logger.success("Training completed.")

        # Evaluate the model
        logger.info("Evaluating model...")
        test_loss, test_accuracy = evaluate_model(model, test_loader)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

        # Save the model
        model.eval()
        model_path = cfg.model.save_path
        torch.save(model.state_dict(), f"{model_path}/model.pth")
        with open(f"{model_path}/model.pkl", "wb") as file:
            pickle.dump(f"{model_path}/model.pth", file)

        # Save the model as a .onnx file
        onnx_model_path = f"{model_path}/model.onnx"
        dummy_input = torch.randn(1, input_size)  # Create a dummy input for the export
        torch.onnx.export(
            model,  # The trained model
            dummy_input,  # Dummy input tensor
            onnx_model_path,  # File path to save the .onnx model
            export_params=True,  # Store the trained parameter weights
            input_names=["input"],  # Input tensor names
            output_names=["output"],  # Output tensor names
            dynamic_axes={  # Dynamic axes for batch size
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=11,  # ONNX opset version (ensure compatibility)
        )
        logger.info(f"Model exported as ONNX to {onnx_model_path}")
        wandb.save(model_path)  # Save the model artifact to W&B

        artifact_filepath = model_path

        logged_artifact = run.log_artifact(
            artifact_filepath, "artifact-name", type="model"
        )
        run.link_artifact(
            artifact=logged_artifact,
            target_path="dtu_mlops_24/wandb-registry-ExamMLOps/test_collection",
        )

        logger.info(f"Model saved to {model_path}")

        # Predict sector for a new company
        logger.info("Predicting sector for a new company...")
        new_company_transformed = preprocess_new_company(
            cfg.api.new_company, column_transformer
        )
        sector_index = predict_sector(model, new_company_transformed)
        sector_name = pd.get_dummies(pd.read_csv(file_path)["Sector"]).columns[
            sector_index
        ]
        wandb.log({"predicted_sector": sector_name})
        logger.success(f"Predicted Sector: {sector_name}")

    except Exception:
        logger.exception("An error occurred during execution.")
        wandb.finish(exit_code=1)


if __name__ == "__main__":
    main()
