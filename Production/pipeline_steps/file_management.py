import os
import shutil
import pandas as pd

def move_new_runs_by_labels():
    src_dir = os.path.join(os.path.dirname(__file__), '../new_runs')
    labeled_dir = os.path.join(os.path.dirname(__file__), '../labeled_runs')
    unlabeled_dir = os.path.join(os.path.dirname(__file__), '../unlabeled_runs')
    labels_dir = os.path.join(os.path.dirname(__file__), '../new_labels')
    app_resources_labels = os.path.join(os.path.dirname(__file__), '../app_resources/labels.csv')
    os.makedirs(labeled_dir, exist_ok=True)
    os.makedirs(unlabeled_dir, exist_ok=True)
    # Find the first CSV in new_labels, ignore .gitkeep
    csv_files = [f for f in os.listdir(labels_dir) if f.endswith('.csv') and f != '.gitkeep']
    if len(csv_files) > 1:
        raise RuntimeError('Error: More than one file found in new_labels. Only one label CSV is allowed.')
    if not csv_files:
        print('No label.csv provided in new_labels. Moving all new_runs to unlabeled_runs.')
        for filename in os.listdir(src_dir):
            if filename == '.gitkeep':
                continue
            src_file = os.path.join(src_dir, filename)
            if not os.path.isfile(src_file):
                continue
            dst_file = os.path.join(unlabeled_dir, filename)
            shutil.move(src_file, dst_file)
            print(f"Moved {filename} to unlabeled_runs.")
        return
    labels_csv = os.path.join(labels_dir, csv_files[0])
    df = pd.read_csv(labels_csv)
    # Ensure only two columns: file_name and label
    if list(df.columns) != ["file_name", "label"]:
        raise ValueError(f"Label CSV must have exactly two columns: 'file_name' and 'label'. Found columns: {list(df.columns)}")
    # Append new labels to app_resources/labels.csv
    if os.path.exists(app_resources_labels):
        existing_labels = pd.read_csv(app_resources_labels)
        combined_labels = pd.concat([existing_labels, df], ignore_index=True).drop_duplicates(subset=["file_name"], keep="last")
    else:
        combined_labels = df
    combined_labels.to_csv(app_resources_labels, index=False)
    print(f"New labels have been stored in app_resources/labels.csv. You may now remove the label csv from the new_labels folder.")
    labeled_files = set(df['file_name'].astype(str))
    for filename in os.listdir(src_dir):
        if filename == '.gitkeep':
            continue
        src_file = os.path.join(src_dir, filename)
        if not os.path.isfile(src_file):
            continue
        if filename in labeled_files:
            dst_file = os.path.join(labeled_dir, filename)
            shutil.move(src_file, dst_file)
            print(f"Moved {filename} to labeled_runs.")
        else:
            dst_file = os.path.join(unlabeled_dir, filename)
            shutil.move(src_file, dst_file)
            print(f"Moved {filename} to unlabeled_runs.")