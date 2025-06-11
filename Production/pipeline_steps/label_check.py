import os
import pandas as pd

def check_labels_and_runs():
    labels_dir = os.path.join(os.path.dirname(__file__), '../New_Labels')
    runs_dir = os.path.join(os.path.dirname(__file__), '../All_Runs')
    csv_files = [f for f in os.listdir(labels_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError('No CSV file found in New_Labels directory.')
    labels_csv = os.path.join(labels_dir, csv_files[0])
    df = pd.read_csv(labels_csv)
    num_labels = len(df)
    num_runs = len([f for f in os.listdir(runs_dir) if os.path.isfile(os.path.join(runs_dir, f))])
    if num_labels != num_runs:
        raise ValueError(f'Length mismatch: {num_labels} labels vs {num_runs} runs in All_Runs.')
    print(f'Label count matches run count: {num_labels}')
