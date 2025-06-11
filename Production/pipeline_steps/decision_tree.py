import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return clf

def plot_and_save_tree(clf, feature_names, class_names, png_path):
    plt.figure(figsize=(40, 20))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=12)
    plt.title("Decision Tree Classifier", fontsize=20)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Tree saved to: {png_path}")
    plt.close()

def retrain_decision_tree(merged_features, model_out, tree_png):
    y = merged_features['label']
    X_clean = merged_features.drop(columns=["file_name", "label"])
    clf = train_and_evaluate(X_clean, y)
    joblib.dump(clf, model_out)
    print(f"Model saved to: {model_out}")
    plot_and_save_tree(clf, X_clean.columns, [str(cls) for cls in sorted(y.unique())], tree_png)
