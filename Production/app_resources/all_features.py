import sys
from pathlib import Path
# Add pipeline_steps to sys.path for direct script execution
pipeline_steps_path = Path(__file__).resolve().parent.parent / 'pipeline_steps'
if str(pipeline_steps_path) not in sys.path:
    sys.path.insert(0, str(pipeline_steps_path))
from feature_extraction import extract_all_features
import pandas as pd
import time

def extract_all_features_to_csv():
    base_dir = Path(__file__).parent.parent
    data_dirs = [base_dir / "labeled_runs", base_dir / "unlabeled_runs"]
    all_files = []
    for dir_path in data_dirs:
        all_files.extend([str(f) for f in dir_path.glob('*.csv')])
    extracted_features = []
    for f in all_files:
        start = time.time()
        features = extract_all_features(Path(f))
        # Store only the child file name (no parent folders)
        features["file_name"] = Path(features["file_name"]).name
        extracted_features.append(features)
    feature_df = pd.DataFrame(extracted_features)
    feature_df.fillna(0, inplace=True)
    output_path = Path(__file__).parent / "features.csv"
    feature_df.to_csv(output_path, index=False)
    print("Features saved successfully to 'features.csv'!")

if __name__ == "__main__":
    extract_all_features_to_csv()
