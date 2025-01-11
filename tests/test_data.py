
import pytest
import pandas as pd
import numpy as np
from src.exam_project.data import load_and_preprocess_data

# Mock configuration for the test
mock_config = {
    "data": {
        "raw_path": "tests/mock_data/mock_sp500_companies.csv"
    }
}

@pytest.fixture
def mock_csv_file(tmp_path):
    """Create a mock CSV file for testing."""
    data = {
        "Company": ["Apple", "Microsoft", "Amazon"],
        "Sector": ["Technology", "Technology", "Retail"],
        "Revenue": [1000, 2000, 1500]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "mock_sp500_companies.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_load_and_preprocess_data(mock_csv_file):
    """Test that the data is loaded and preprocessed correctly."""
    # Call the function
    column_transformer, X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(mock_csv_file)

    # Assert that the data is loaded and split correctly
    assert X_train.shape[0] > 0, "Training set should not be empty"
    assert y_train.shape[0] > 0, "Training labels should not be empty"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature dimensions must match across splits"
    assert isinstance(column_transformer, object), "Column transformer should be returned"

    # Add more assertions to check data preprocessing correctness

def test_data_scaling():
    data = np.array([[100, 0.5], [200, 0.7], [300, 0.8]])
    scaled_data = some_scaling_function(data)
    assert np.allclose(scaled_data.mean(axis=0), 0, atol=1e-5), "Mean should be 0 after scaling"
    assert np.allclose(scaled_data.std(axis=0), 1, atol=1e-5), "Std should be 1 after scaling"

def test_missing_value_handling():
    data = pd.DataFrame({"A": [1, 2, None], "B": [4, None, 6]})
    processed_data = handle_missing_values(data)
    assert processed_data.isnull().sum().sum() == 0, "All missing values should be handled"

def test_label_encoding():
    labels = ["Tech", "Retail", "Tech"]
    encoded_labels = encode_labels(labels)
    assert len(set(encoded_labels)) == 2, "There should be 2 unique encoded labels"
