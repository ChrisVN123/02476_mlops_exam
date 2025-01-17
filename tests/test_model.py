import pytest
import torch

from src.exam_project.model import SectorClassifier


# Fixture to create a test model
@pytest.fixture
def test_model():
    input_size = 10  # Example input size
    num_classes = 5  # Example number of classes
    return SectorClassifier(input_size, num_classes)


def test_model_forward_pass(test_model):
    """Test if the model produces the correct output shape."""
    input_size = 10
    num_classes = 5
    batch_size = 16

    # we just wish to tes the model forward pass so we get a random input
    test_input = torch.randn(batch_size, input_size)

    # we perform a forward pass to test for correctness
    output = test_model(test_input)

    # now check the shape of the output
    assert output.shape == (batch_size, num_classes), (
        f"Expected output shape {(batch_size, num_classes)}, " f"but got {output.shape}"
    )


def test_model_training_step(test_model):
    """Test if the model can compute gradients during training."""
    input_size = 10
    num_classes = 5
    batch_size = 16

    # create random input and target tensors
    test_input = torch.randn(batch_size, input_size)
    test_target = torch.randint(0, num_classes, (batch_size,))

    # define the loss functon and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(test_model.parameters())

    # perform a forward pass
    output = test_model(test_input)

    # compute loss
    loss = criterion(output, test_target)

    # perform backward pass
    optimizer.zero_grad()
    loss.backward()

    # check if gradients are computed
    for param in test_model.parameters():
        assert param.grad is not None, "Gradient not computed for model parameters"


def test_model_save_and_load(test_model, tmp_path):
    """Test saving and loading the model."""
    model_path = tmp_path / "test_model.pth"

    # save the model
    torch.save(test_model.state_dict(), model_path)

    # re-create the model with the same architecture
    input_size = test_model.fc1.in_features  # Input size from the saved model
    num_classes = (
        test_model.fc3.out_features
    )  # Number of output classes from the saved model
    loaded_model = SectorClassifier(input_size, num_classes)

    # load the saved state into the new model
    loaded_model.load_state_dict(torch.load(model_path))

    # ensure the loaded model works as expected
    loaded_model.eval()
    test_input = torch.randn(1, input_size)  # Test input with correct size
    output = loaded_model(test_input)

    assert output.shape == (
        1,
        num_classes,
    ), "Loaded model's output shape is incorrect"
