import os
import subprocess  # To run DVC commands
import yaml
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore

from src.exam_project.__init__ import (_PATH_DATA) # type: ignore

def load_and_preprocess_data(file_path: str = "data/raw/sp500_companies.csv", wandb_run=None, work_dir = None):
    data = pd.read_csv(file_path)

    try:
        # 1. Add the file to DVC
        print(f"Tracking file {file_path} with DVC...")
        subprocess.run(["dvc", "add", file_path], check=True)
        subprocess.run(["dvc", "push"], check=True)
        print(f"File {file_path} successfully pushed to remote storage.")
        
        # 2. Read the generated .dvc file
        dvc_file_path = f"{work_dir}/data.dvc"  # Ensure it matches what DVC actually created
        if not os.path.isfile(dvc_file_path):
            raise FileNotFoundError(f"Could not find DVC metafile: {dvc_file_path}")
        
        with open(dvc_file_path, "r") as f:
            dvc_content = yaml.safe_load(f)
        print(f"DVC content: {dvc_content}")


        # 4. Optional: log to wandb
        if wandb_run:
            wandb_run.log({"raw_data_path": file_path, "dvc_content": dvc_content})
            dvc_file_path = os.path.join(wandb_run.dir, "dvc_content.yaml")
            with open(dvc_file_path, "w") as f:
                yaml.safe_dump(dvc_content, f)





    except subprocess.CalledProcessError as e:
        print(f"Error running DVC command: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


    
        
    data = data.drop(
    columns=[
        "Longbusinesssummary",
        "City",
        "State",
        "Country",
        "Shortname",
        "Longname",
    ]
    )
    data = data.dropna()
    # Log dataset metadata if wandb_run is active

    X = data.drop(columns=["Sector"])
    y = data["Sector"]

    categorical_features = ["Exchange", "Symbol", "Industry"]
    numerical_features = [
        "Currentprice",
        "Marketcap",
        "Ebitda",
        "Revenuegrowth",
        "Fulltimeemployees",
        "Weight",
    ]

    column_transformer = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    X_transformed = column_transformer.fit_transform(X)
    y_encoded = pd.get_dummies(y).values
    sector_names = pd.get_dummies(y).columns.tolist()  # Get the sector names
    pd.DataFrame(sector_names, columns=["Sector"]).to_csv(
        f"{_PATH_DATA}/processed/sector_names.csv", index=False
    )
    pd.DataFrame(sector_names, columns=["Sector"]).to_csv(
        f"{_PATH_DATA}/processed/sector_names.csv", index=False
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_transformed,
        y_encoded,
        test_size=0.3,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
    )

    # Ensure the processed directory exists
    processed_dir = f"{_PATH_DATA}/processed"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    pd.DataFrame(X_test).to_csv(f"{_PATH_DATA}/processed/X_test.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{_PATH_DATA}/processed/y_test.csv", index=False)
    return (
        column_transformer,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        sector_names,
    )


def find_sparse_rows_by_initials(X, y, column_transformer, initials: str = "NOW"):
    """
    Find rows in the sparse matrix X where the column 'cat__Symbol_{initials}' equals 1.

    Args:
        X: The sparse matrix (e.g., X_test).
        column_transformer: Fitted ColumnTransformer to retrieve column names.
        initials: The initials (e.g., "NOW") to search for in the 'cat__Symbol_{initials}' column.

    Returns:
        List of rows in sparse matrix format that match the criteria.
    """
    # Generate the target column name
    target_column = f"cat__Symbol_{initials}"

    # Get all column names from the ColumnTransformer
    column_names = column_transformer.get_feature_names_out()
    # Filter to get columns matching "cat__Symbol_"
    symbol_columns = [name for name in column_names if name.startswith("cat__Symbol_")]

    # Dynamically filter symbol_columns to only include those with at least one non-zero value
    valid_symbol_columns = []
    for col_name in symbol_columns:
        col_index = np.where(column_names == col_name)[0][0]
        if X[:, col_index].nnz > 0:  # nnz gives the number of non-zero entries

            valid_symbol_columns.append(col_name)
    # Find the index of the target column
    if target_column not in valid_symbol_columns:
        # Extract available initials
        available_initials = [
            name.split("cat__Symbol_")[1] for name in valid_symbol_columns
        ]

        # Raise the ValueError first
        raise ValueError(
            f"{initials}' not found in company list.\n"
            f"Available initials to choose from:\n {', '.join(available_initials)}"
        )

    target_column_index = list(column_names).index(target_column)
    row_index = X[:, target_column_index].nonzero()[0][0]
    y_index = y[row_index].nonzero()[0][0]

    return X[row_index].toarray().astype(np.float32), y_index


if __name__ == "__main__":
    column_transformer, X_train, X_val, X_test, y_train, y_val, y_test, sector_names = (
        load_and_preprocess_data()
    )
    X, y = find_sparse_rows_by_initials(X_test, y_test, column_transformer, "AEE")
    print(X[0], y)
