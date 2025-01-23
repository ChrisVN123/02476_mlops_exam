
import pandas as pd
from sklearn.model_selection import train_test_split

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


file_path = "data/raw/sp500_companies.csv"
data = pd.read_csv(file_path)

reference_data, current_data = train_test_split(data, test_size=0.3, random_state=42)


report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

report.save_html('report.html')
