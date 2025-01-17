import os

import matplotlib.pyplot as plt
import pandas as pd
import typer


def dataset_statistics(data_path: str = "data/raw/sp500_companies.csv") -> None:
    """Compute basic dataset statistics and checks, including other files in the directory."""

    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Check for other files in the data/raw/ directory
    raw_data_dir = os.path.dirname(data_path)
    raw_files = os.listdir(raw_data_dir)
    print(f"\nFiles in {raw_data_dir}:")
    print(raw_files)

    # Ensure only the expected file exists
    if len(raw_files) > 1:
        print("Warning: Other files found in the directory. Ensure they are intended:")
        for file in raw_files:
            if file != os.path.basename(data_path):
                print(f" - {file}")

    # Load the dataset
    print(f"\nLoading dataset from: {data_path}")
    data = pd.read_csv(data_path)

    # General dataset checks
    print(f"\nDataset file name: {os.path.basename(data_path)}")
    print(f"Number of rows: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")
    print(f"Columns: {list(data.columns)}")

    # Check for missing values
    missing_values = data.isnull().sum().sum()
    print(f"\nTotal missing values: {missing_values}")

    if missing_values > 0:
        print("Columns with missing values:")
        print(data.isnull().sum()[data.isnull().sum() > 0])

    # Sample data preview
    print("\nSample of the data:")
    print(data.head())

    # Distribution of a categorical column (e.g., Sector)
    if "Sector" in data.columns:
        sector_counts = data["Sector"].value_counts()
        print("\nSector distribution:")
        print(sector_counts)
        sector_counts.plot(kind="bar", title="Sector Distribution")
        plt.xlabel("Sector")
        plt.ylabel("Count")
        plt.savefig("results/sector_distribution.png")
        plt.close()
        print("Sector distribution plot saved as 'sector_distribution.png'.")

    # Check dataset size consistency
    expected_columns = [
        "Exchange",
        "Symbol",
        "Industry",
        "Currentprice",
        "Marketcap",
        "Sector",
    ]
    if all(column in data.columns for column in expected_columns):
        print("\nAll expected columns are present.")
    else:
        missing_columns = [col for col in expected_columns if col not in data.columns]
        print(f"\nMissing columns: {missing_columns}")


if __name__ == "__main__":
    typer.run(dataset_statistics)
