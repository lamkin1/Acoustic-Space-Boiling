import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_and_merge_data(feature_path, label_path):
    # Load features
    X = pd.read_csv(feature_path)
    X['file_name'] = X['file_name'].str.replace("Data/After_May/MATLAB ", "", regex=False).str.strip().str.lower()

    # Load labels
    y_df = pd.read_csv(label_path)
    y_df['file_name'] = y_df['file_name'].str.strip().str.lower()

    # Merge
    df = pd.merge(X, y_df, on="file_name", how="inner")
    print(f"Merged rows: {df.shape[0]}")
    return df

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

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

def main():
    parser = argparse.ArgumentParser(description="Train and visualize a Decision Tree Classifier.")
    parser.add_argument("feature_csv", help="Path to features CSV file")
    parser.add_argument("labels_csv", help="Path to labels CSV file")
    parser.add_argument("model_out", help="Filename to serialize the trained model (e.g., model.pkl)")
    parser.add_argument("tree_png", help="Filename to save decision tree PNG (e.g., tree.png)")
    args = parser.parse_args()

    df = load_and_merge_data(args.feature_csv, args.labels_csv)
    y = df['label']
    X_clean = df.drop(columns=["file_name", "label"])

    clf = train_and_evaluate(X_clean, y)
    joblib.dump(clf, args.model_out)
    print(f"Model saved to: {args.model_out}")

    # plot_feature_importances(clf, X_clean.columns)
    plot_and_save_tree(clf, X_clean.columns, [str(cls) for cls in sorted(y.unique())], args.tree_png)

if __name__ == "__main__":
    main()
