import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import graphviz

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


def retrain_decision_tree(merged_features, model_out, tree_png, tree_svg=None):
    X_clean = merged_features.drop(columns=["file_name", "label"])
    

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(merged_features['label'])
    clf = train_and_evaluate(X_clean, pd.Series(y_encoded))
    joblib.dump(clf, model_out)
    print(f"Model saved to: {model_out}")
    plot_and_save_tree(clf, X_clean.columns, list(map(str, label_encoder.classes_)), tree_png)
    # Always recreate the SVG in app_resources/output
    from pathlib import Path
    output_dir = Path(__file__).parent.parent / "app_resources" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    tree_svg_path = output_dir / "tree.svg"
    save_tree_as_svg(clf, X_clean.columns, list(map(str, label_encoder.classes_)), tree_svg_path)
    print(f"Tree SVG saved to: {tree_svg_path}")
    # Always recreate the dtreeviz SVG in app_resources/output
    dtreeviz_svg_path = str(output_dir / "dtreeviz.svg")
    plot_and_save_dtreeviz(clf, X_clean, y_encoded, list(X_clean.columns), list(map(str, label_encoder.classes_)), dtreeviz_svg_path)
    print(f"dtreeviz SVG saved to: {dtreeviz_svg_path}")
