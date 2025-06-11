import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import export_graphviz
import graphviz
import dtreeviz
import os

def load_and_merge_data(feature_path, label_path):
    X = pd.read_csv(feature_path)
    print(X.columns)
    X['file_name'] = (
        X['file_name']
        .apply(lambda x: os.path.basename(str(x)))
        .str.strip()
        .str.lower()
    )

    y_df = pd.read_csv(label_path)
    print(y_df.columns)
    y_df['file_name'] = (
        y_df['file_name']
        .str.replace("MATLAB ", "", regex=False)
        .str.replace(".png", "", regex=False)
        .str.strip()
        .str.lower()
    )

    df = pd.merge(X, y_df, on="file_name", how="inner")

    if df.shape[0] == 0:
        print("Sample features file names:", X['file_name'].head())
        print("Sample labels file names:", y_df['file_name'].head())
        raise ValueError("Merge resulted in 0 rows. Check file_name formatting.")

    print(f"Merged rows: {df.shape[0]}")
    return df

def train_and_evaluate(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)

    class_labels = np.unique(y)

    class_sums = {label: {'precision_sum': 0.0, 'recall_sum': 0.0, 'f1_sum': 0.0, 'support': 0} for label in class_labels}

    macro_metrics = {'precision': [], 'recall': [], 'f1': []}
    weighted_numerator = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    total_support = 0

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        p_class, r_class, f_class, s_class = precision_recall_fscore_support(
            y_test, y_pred, labels=class_labels, average=None, zero_division=0
        )

        for i, label in enumerate(class_labels):
            class_sums[label]['precision_sum'] += p_class[i] * s_class[i]
            class_sums[label]['recall_sum'] += r_class[i] * s_class[i]
            class_sums[label]['f1_sum'] += f_class[i] * s_class[i]
            class_sums[label]['support'] += s_class[i]

        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        macro_metrics['precision'].append(p_macro)
        macro_metrics['recall'].append(r_macro)
        macro_metrics['f1'].append(f_macro)

        p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        support_fold = sum(s_class)
        weighted_numerator['precision'] += p_weighted * support_fold
        weighted_numerator['recall'] += r_weighted * support_fold
        weighted_numerator['f1'] += f_weighted * support_fold
        total_support += support_fold

    per_class_summary = []
    for label in class_labels:
        support = class_sums[label]['support']
        if support == 0:
            precision = recall = f1 = 0.0
        else:
            precision = class_sums[label]['precision_sum'] / support
            recall = class_sums[label]['recall_sum'] / support
            f1 = class_sums[label]['f1_sum'] / support

        per_class_summary.append({
            'Class': label,
            'Support': support,
            'Weighted Precision': precision,
            'Weighted Recall': recall,
            'Weighted F1-Score': f1
        })

    per_class_df = pd.DataFrame(per_class_summary)

    summary_df = pd.DataFrame({
        'Average Type': ['Macro', 'Weighted'],
        'Precision': [
            np.mean(macro_metrics['precision']),
            weighted_numerator['precision'] / total_support
        ],
        'Recall': [
            np.mean(macro_metrics['recall']),
            weighted_numerator['recall'] / total_support
        ],
        'F1-Score': [
            np.mean(macro_metrics['f1']),
            weighted_numerator['f1'] / total_support
        ]
    })

    print("\n=== Weighted Per-Class Average Metrics Across Folds ===")
    print(per_class_df.to_string(index=False))

    print("\n=== Overall Macro and Weighted Averages ===")
    print(summary_df.to_string(index=False))

    return clf

def plot_feature_importances(clf, feature_names):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances from Decision Tree", fontsize=16)
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), feature_names[indices], rotation=45, ha='right')
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_and_save_tree(clf, feature_names, class_names, png_path):
    plt.figure(figsize=(40, 20))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=12)
    plt.title("Decision Tree Classifier", fontsize=20)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Tree saved to: {png_path}")
    plt.close()


def save_tree_as_svg(clf, feature_names, class_names, out_svg_path):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=[str(cls) for cls in class_names],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    svg_content = graph.pipe(format='svg').decode('utf-8')
    with open(out_svg_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

def plot_and_save_dtreeviz(clf, X, y, feature_names, class_names, out_svg_path):
    from dtreeviz import model
    viz = model(
        clf,
        X,
        y,
        target_name="label",
        feature_names=feature_names,
        class_names=[str(cls) for cls in class_names]
    )
    viz.view().save(out_svg_path)
    print(f"dtreeviz visualization saved to: {out_svg_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and visualize a Decision Tree Classifier.")
    parser.add_argument("feature_csv", help="Path to features CSV file")
    parser.add_argument("labels_csv", help="Path to labels CSV file")
    parser.add_argument("model_out", help="Filename to serialize the trained model (e.g., model.pkl)")
    parser.add_argument("tree_png", help="Filename to save decision tree PNG (e.g., tree.png)")
    parser.add_argument("tree_svg", help="Filename to save decision tree SVG (e.g., tree.svg)")
    args = parser.parse_args()

    df = load_and_merge_data(args.feature_csv, args.labels_csv)
    print("Sample file names from features CSV:", df['file_name'].head())
    print("Sample file names from labels CSV:", df['file_name'].head())

    y = df['label']
    from sklearn.preprocessing import LabelEncoder

    # Encode labels as integers starting from 0
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['label'])

    X_clean = df.drop(columns=["file_name", "label"])
    clf = train_and_evaluate(X_clean, pd.Series(y_encoded))

    # Save model
    joblib.dump(clf, args.model_out)
    print(f"Model saved to: {args.model_out}")

    # Visualizations
   # plot_feature_importances(clf, X_clean.columns)
    plot_and_save_tree(clf, X_clean.columns, list(map(str, label_encoder.classes_)), args.tree_png)
    save_tree_as_svg(clf, X_clean.columns, label_encoder.classes_, args.tree_svg)
    plot_and_save_dtreeviz(clf, X_clean, y_encoded, list(X_clean.columns), label_encoder.classes_, "dtreeviz.svg")

if __name__ == "__main__":
    main()
