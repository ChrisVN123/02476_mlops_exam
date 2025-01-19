import hydra
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.2")
def dataset_statistics(cfg: DictConfig, report_path: str = "report.md") -> None:
    """
    Generate dataset statistics and save to a markdown report.

    Args:
        report_path (str): Path to save the generated markdown report.
    """
    # Load the dataset
    data_path = cfg.data.raw_path
    df = pd.read_csv(data_path)

    # Open a markdown file to write the report
    with open(report_path, "w") as report:
        report.write("# Dataset Statistics\n\n")
        report.write(f"**Dataset**: `{data_path}`\n\n")
        report.write(f"**Number of Rows**: {len(df)}\n\n")
        report.write(f"**Number of Columns**: {len(df.columns)}\n\n")
        report.write("**Columns**:\n")
        for col in df.columns:
            report.write(f"- {col}\n")
        report.write("\n")

        # Generate Sector Distribution Visualization
        if "Sector" in df.columns:
            plt.figure(figsize=(10, 5))
            df["Sector"].value_counts().plot(kind="bar")
            plt.title("Sector Distribution")
            plt.xlabel("Sector")
            plt.ylabel("Count")
            plt.savefig("sector_distribution.png")
            plt.close()
            report.write("### Sector Distribution\n")
            report.write("![Sector Distribution](./sector_distribution.png)\n\n")
        else:
            report.write("### Sector Distribution\n")
            report.write("Sector column not found in the dataset.\n\n")

        # Summary of missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            report.write("### Missing Values\n")
            report.write("The following columns have missing values:\n")
            for col, num_missing in missing_values.items():
                if num_missing > 0:
                    report.write(f"- {col}: {num_missing} missing values\n")
            report.write("\n")
        else:
            report.write("### Missing Values\n")
            report.write("No missing values in the dataset.\n\n")

        # File size information
        file_size = df.memory_usage(deep=True).sum() / (1024**2)  # Convert bytes to MB
        report.write(f"**Dataset File Size**: {file_size:.2f} MB\n")


if __name__ == "__main__":
    dataset_statistics()
