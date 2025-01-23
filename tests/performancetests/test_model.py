import os
import time
import torch
import wandb
from src.exam_project.model import SectorClassifier  # Replace with your model class

def load_model():
    """
    Load the model from a W&B artifact.
    """

    logdir = "./artifacts"  # Define the directory for downloading artifacts

    # Initialize W&B API
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={
            "entity": os.getenv("WANDB_ENTITY"),
            "project": os.getenv("WANDB_PROJECT"),
        },
    )

    # Fetch and download the artifact
    artifact = api.artifact("test_collection:v3")
    artifact.download(root=logdir)

    # Load the model checkpoint
    file_name = artifact.files()[0].name
    model = SectorClassifier.load_from_checkpoint(f"{logdir}/{file_name}")
    return model

def test_model_speed():
    """
    Test the inference speed of the model.
    """
    model = load_model()
    model.eval()  # Ensure the model is in evaluation mode

    input_tensor = torch.rand(1, 561)  # Adjust shape based on your model's input requirements
    start = time.time()
    for _ in range(100):
        _ = model(input_tensor)  # Run inference
    end = time.time()

    # Ensure the test runs within a reasonable time
    assert end - start < 1, "Inference time exceeds 1 second for 100 runs."

if __name__ == "__main__":
    test_model_speed()
    print("Model speed test passed!")
