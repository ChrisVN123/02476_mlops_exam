import numpy as np
from sklearn.compose import ColumnTransformer

from exam_project.data import load_and_preprocess_data
from tests.__init__ import \
    _PATH_DATA  # Adjust this import to your actual utils module

# new change


def test_load_and_preprocess_data():
    """Test that the data is loaded and preprocessed correctly."""
    # Load and preprocess the data
    column_transformer, X_train, X_val, X_test, y_train, y_val, y_test = (
        load_and_preprocess_data(f"{_PATH_DATA}/sp500_companies.csv")
    )

    # Assert that the data is loaded and split correctly
    assert X_train.shape[0] > 0, "Training set should not be empty"
    assert y_train.shape[0] > 0, "Training labels should not be empty"
    assert (
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    ), "Feature dimensions must match across splits"
    assert isinstance(
        column_transformer, ColumnTransformer
    ), "Column transformer should be returned"

    # Convert to dense array if sparse
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()

    # Assert that all missing values are handled
    assert not np.isnan(X_train).any(), "X_train contains missing values"
    assert not np.isnan(X_val).any(), "X_val contains missing values"
    assert not np.isnan(X_test).any(), "X_test contains missing values"

    # Check label dimensions
    assert (
        y_train.shape[1] == y_val.shape[1] == y_test.shape[1]
    ), "Label dimensions must match across splits"
