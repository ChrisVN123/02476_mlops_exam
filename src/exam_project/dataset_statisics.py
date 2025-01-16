import pandas as pd
import matplotlib.pyplot as plt
import typer


def dataset_statistics(dataset_path: str = "data/raw/sp500_companies.csv") -> None:
    """Generate basic statistics and figures for the dataset."""
    print("Generating dataset statistics...")

    data = pd.read_csv(dataset_path)

    print(f"Number of samples: {len(data)}")
    print(f"Columns: {list(data.columns)}")

    if 'Sector' in data.columns:
        class_distribution = data['Sector'].value_counts()
        print("Class Distribution:")
        for cls, count in class_distribution.items():
            print(f"  {cls}: {count}")

    print(type(class_distribution))

    plt.figure(figsize=(10, 6)) 
    plt.bar(class_distribution.index, class_distribution.values)

    plt.xticks(rotation=45, ha='right')
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()

if __name__ == "__main__":
    typer.run(dataset_statistics)
