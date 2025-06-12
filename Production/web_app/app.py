import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import io
import base64
import time
import json
import os
import numpy as np
import pickle
import subprocess
import tempfile
from pathlib import Path
import importlib.util
import joblib
from sklearn.preprocessing import StandardScaler

# Add this near the top of the file with other constants
REGIME_MAPPING = {
    0: "Single Rhythmic",
    1: "Double Rhythmic",
    2: "Random",
    3: "Rhythmic with Climax",
    4: "Noise",
    5: "1 Rhythmic + Random",
    6: "Triple Rhythmic",
    7: "Transition"
}

# --- Helper Functions ---
def get_svg_path():
    """Get the absolute path to the dtreeviz.svg file."""
    # Get the absolute path to the Production directory
    production_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Construct path to the SVG file
    svg_path = os.path.join(production_dir, 'app_resources', 'output', 'dtreeviz.svg')
    return svg_path

def run_feature_extraction(input_file):
    """Run feature extraction for a single file using extract_all_features directly."""
    # Dynamically import extract_all_features from pipeline_steps/feature_extraction.py
    production_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    feature_extraction_path = os.path.join(production_dir, 'pipeline_steps', 'feature_extraction.py')
    spec = importlib.util.spec_from_file_location("feature_extraction", feature_extraction_path)
    fe_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_module)
    extract_all_features = fe_module.extract_all_features

    # Call extract_all_features on the uploaded file
    features = extract_all_features(Path(input_file))
    # Store only the child file name (no parent folders)
    features["file_name"] = Path(features["file_name"]).name
    features_df = pd.DataFrame([features])
    features_df.fillna(0, inplace=True)
    return features_df

def load_tree_model():
    """Load the saved decision tree model."""
    production_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(production_dir, 'app_resources', 'output', 'tree_model.pkl')

    try:
        model = joblib.load(model_path)

        # Verify that the loaded object is a scikit-learn model
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded model is not a valid scikit-learn model")

        print("[DEBUG] Model type:", type(model))
        print("[DEBUG] Model attributes:", dir(model))
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        raise

def get_cluster_data(new_point=None, clusters_count=2):
    """Get clustering data including the new point if provided."""
    production_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_path = os.path.join(production_dir, 'app_resources', 'output', f'results_{clusters_count}_clusters.csv')

    df = pd.read_csv(results_path)
    df["file_name"] = df["file_name"].str[15:]  # Remove 'MATLAB ' prefix if present

    # If a new point is provided, add it to the dataframe
    if new_point:
        new_row = pd.DataFrame({
            'PCA1': [new_point['x']],
            'PCA2': [new_point['y']],
            'PCA3': [new_point['z']],
            'file_name': ['New Run'],
            f'Cluster_{clusters_count}_PCs': [new_point['cluster']]
        })
        df = pd.concat([df, new_row], ignore_index=True)

    return df

def preprocess(data):
    """Scale the data using StandardScaler."""
    string_cols = []
    na_cols = []
    numeric_cols = []
    for col in data.columns:
        if pd.api.types.is_string_dtype(data[col]):
            string_cols.append(col)
        elif data[col].isnull().any():
            na_cols.append(col)
        else:
            numeric_cols.append(col)
    removed_cols = string_cols + na_cols
    if removed_cols:
        print("Removed columns:", removed_cols)
        data = data[numeric_cols]
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(data)
    return scaledData, numeric_cols

def run_ml_pipeline(df):
    """Run the full ML pipeline on the uploaded data."""
    # 1. Feature Extraction
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        input_path = temp_file.name
        df.to_csv(input_path, index=False)

    try:
        # Extract features
        features_df = run_feature_extraction(input_path)
        print("[DEBUG] Extracted features:")
        print(features_df)

        # Load the model
        model = load_tree_model()
        print("[DEBUG] Loaded model:")
        print(model)

        # Ensure features match model's expected input
        feature_names = model.feature_names_in_
        features_for_prediction = features_df[feature_names]

        # Make prediction
        prediction = model.predict(features_for_prediction)[0]
        probabilities = model.predict_proba(features_for_prediction)[0]
        confidence = max(probabilities)
        print(f"[DEBUG] Prediction: {prediction}, Probabilities: {probabilities}, Confidence: {confidence}")

        # Load the saved scaler
        production_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        scaler_path = os.path.join(production_dir, 'app_resources', 'output', 'scaler.pkl')
        scaler = joblib.load(scaler_path)

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(features_df[feature_names])
        scaled_features_df = pd.DataFrame(scaled_features, columns=feature_names)
        print("[DEBUG] Scaled features (using saved scaler):")
        print(scaled_features_df)

        # Apply PCA to get coordinates
        production_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        loadings_path = os.path.join(production_dir, 'app_resources', 'output', 'loadings.csv')
        loadings = pd.read_csv(loadings_path, index_col='Feature')

        # Get PCA coordinates
        pca_coords = {}
        for i in range(1, 7):  # PCA1 through PCA6
            pca_col = f'PCA{i}'
            # Calculate PCA coordinate by multiplying scaled features with loadings
            pca_value = sum(scaled_features_df[feature].iloc[0] * loadings.loc[feature, pca_col]
                          for feature in feature_names)
            print(f"\n[DEBUG] PCA{i} calculation:")
            print(f"Scaled Features: {scaled_features_df.iloc[0].to_dict()}")
            print(f"Loadings: {loadings[pca_col].to_dict()}")
            print(f"Result: {pca_value}")
            pca_coords[pca_col.lower()] = float(pca_value)

        print(f"[DEBUG] PCA coordinates: {pca_coords}")

        # Get cluster assignment
        cluster_data = get_cluster_data()
        # Find the first cluster column that exists
        cluster_cols = [col for col in cluster_data.columns if col.startswith('Cluster_')]
        if not cluster_cols:
            raise ValueError("No cluster columns found in the data")
        cluster_col = cluster_cols[0]  # Use the first available cluster column
        pca_coords['cluster'] = cluster_data[cluster_col].iloc[-1]  # Use the last cluster assignment
        print(f"[DEBUG] Cluster assignment: {pca_coords['cluster']}")

        result = {
            'class': prediction,
            'confidence': confidence,
            'point': {
                'x': pca_coords['pca1'],
                'y': pca_coords['pca2'],
                'z': pca_coords['pca3'],
                'cluster': pca_coords['cluster']
            },
            'analysis': {
                'influential_features': [
                    {'name': name, 'importance': imp, 'value': val}
                    for name, imp, val in zip(feature_names, model.feature_importances_, features_df[feature_names].iloc[0])
                ],
                'probabilities': [
                    {'class': cls, 'probability': prob}
                    for cls, prob in zip(model.classes_, probabilities)
                ]
            }
        }
        print("[DEBUG] Final result:")
        print(result)

        return result
    finally:
        # Clean up the temporary file
        os.unlink(input_path)

# --- Placeholder Functions ---
# These functions simulate the outputs of your actual machine learning scripts.
# You can replace the logic inside them with calls to your own Python methods.

def get_3d_cluster_data(new_point=None, x_axis='PCA1', y_axis='PCA2', z_axis='PCA3', clusters_count=2):
    """
    Generates or loads 3D clustering data with configurable axes and cluster count.
    """
    # Load the clustering results
    production_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_path = os.path.join(production_dir, 'app_resources', 'output', f'results_{clusters_count}_clusters.csv')

    try:
        df = pd.read_csv(results_path)
        df["file_name"] = df["file_name"].str[15:]  # Remove 'MATLAB ' prefix if present

        # Determine which cluster column to use based on the number of PCA components
        pca_count = max(
            int(x_axis.replace('PCA', '')),
            int(y_axis.replace('PCA', '')),
            int(z_axis.replace('PCA', '')) if z_axis else 0
        )
        color_col = f'Cluster_{pca_count}_PCs'
        color_label = 'Cluster'

        # Create the 3D scatter plot
        fig = go.Figure()

        # Add clusters to the plot
        for name, group in df.groupby(color_col):
            fig.add_trace(go.Scatter3d(
                x=group[x_axis],
                y=group[y_axis],
                z=group[z_axis] if z_axis else None,
                mode='markers',
                marker=dict(size=7),
                name=f'{color_label} {name}',
                customdata=group["file_name"],
                hovertemplate="<b>%{customdata}</b><br>" +
                            f"{x_axis}: %{{x:.2f}}<br>" +
                            f"{y_axis}: %{{y:.2f}}" +
                            (f"<br>{z_axis}: %{{z:.2f}}" if z_axis else "") +
                            "<extra></extra>"
            ))

        # If a new point is provided from classification, add it
        if new_point:
            fig.add_trace(go.Scatter3d(
                x=[new_point['x']], y=[new_point['y']], z=[new_point['z']] if z_axis else None,
                mode='markers',
                marker=dict(size=12, color='#FFD700', symbol='cross'),
                name='New Data Point'
            ))

        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            scene=dict(
                xaxis=dict(showbackground=False, title=x_axis),
                yaxis=dict(showbackground=False, title=y_axis),
                zaxis=dict(showbackground=False, title=z_axis) if z_axis else None,
                bgcolor='#f8fafc',
            ),
            legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, bgcolor='rgba(255,255,255,0.6)')
        )

        # Add camera rotation animation
        frames = []
        angles = np.linspace(0, 360, 120, endpoint=False)
        for i, angle in enumerate(angles):
            theta = np.radians(angle)
            eye = dict(
                x=1.8 * np.sin(theta),
                y=1.8 * np.cos(theta),
                z=0.5
            )
            frames.append(dict(
                name=str(i),
                layout=dict(scene_camera=dict(eye=eye))
            ))

        fig.frames = frames
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=0,
                    x=0,
                    xanchor='left',
                    yanchor='top',
                    pad=dict(t=10, r=10, b=10, l=10),
                    buttons=[
                        dict(
                            label='▶︎ Rotate',
                            method='animate',
                            args=[None,
                                  dict(frame=dict(duration=100, redraw=True),
                                       transition=dict(duration=0),
                                       fromcurrent=True,
                                       mode='immediate')]
                        )
                    ]
                )
            ]
        )

        return fig

    except Exception as e:
        print(f"Error loading clustering data: {str(e)}")
        # Return a basic figure if there's an error
        return go.Figure()

def create_interactive_svg_viewer():
    """
    Creates an interactive SVG viewer with zoom and pan functionality.
    """
    svg_path = get_svg_path()
    # Read the SVG file content
    with open(svg_path, 'r') as f:
        svg_content = f.read()

    # Encode the SVG content to base64
    svg_base64 = base64.b64encode(svg_content.encode()).decode()
    svg_data_url = f"data:image/svg+xml;base64,{svg_base64}"

    return html.Div(
        className="relative w-full h-full border border-gray-200 rounded-lg overflow-hidden bg-white",
        style={'height': '500px'},
        children=[
            # SVG Container with pan and zoom
            html.Div(
                id="svg-container",
                className="w-full h-full overflow-hidden relative cursor-move",
                style={
                    'height': '500px',
                    'backgroundColor': '#ffffff'
                },
                children=[
                    html.Div(
                        id="svg-content",
                        className="transform-gpu transition-transform duration-75",
                        style={
                            'transform-origin': 'center center',
                            'position': 'absolute',
                            'left': '50%',
                            'top': '50%',
                            'width': 'fit-content',
                            'height': 'fit-content',
                            'translate': '-50% -50%'  # tailwind-style centering
                            # ❌ Do NOT add 'transform' here
                        },
                        children=[
                            html.Img(
                                src=svg_data_url,  # Use the base64 encoded SVG data URL
                                alt="Decision Tree Visualization",
                                style={
                                    'max-width': 'none',
                                    'max-height': 'none',
                                    'display': 'block',
                                    'user-select': 'none',
                                    'pointer-events': 'none'
                                }
                            )
                        ]
                    )
                ]
            ),
            # Control Panel
            html.Div(
                className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg shadow-lg p-2 flex flex-col gap-2",
                children=[
                    html.Button(
                        html.I(className="fas fa-plus"),
                        id="zoom-in-btn",
                        className="p-2 hover:bg-gray-100 rounded text-gray-700 transition-colors",
                        title="Zoom In"
                    ),
                    html.Button(
                        html.I(className="fas fa-minus"),
                        id="zoom-out-btn",
                        className="p-2 hover:bg-gray-100 rounded text-gray-700 transition-colors",
                        title="Zoom Out"
                    ),
                    html.Button(
                        html.I(className="fas fa-expand-arrows-alt"),
                        id="reset-view-btn",
                        className="p-2 hover:bg-gray-100 rounded text-gray-700 transition-colors",
                        title="Reset View"
                    ),
                    html.Button(
                        html.I(className="fas fa-download"),
                        id="download-svg-btn",
                        className="p-2 hover:bg-gray-100 rounded text-gray-700 transition-colors",
                        title="Download SVG"
                    )
                ]
            ),
            # Zoom level indicator
            html.Div(
                id="zoom-indicator",
                className="absolute bottom-4 left-4 bg-white/90 backdrop-blur-sm rounded-lg shadow-lg px-3 py-1 text-sm text-gray-700",
                children="100%"
            )
        ]
    )

def get_feature_analysis_data():
    """
    Placeholder to generate feature analysis data.
    """
    return {
        'influential_features': [
            {'name': 'Feature_A', 'importance': 0.34, 'value': 0.72},
            {'name': 'Feature_B', 'importance': 0.28, 'value': -0.15},
            {'name': 'Feature_C', 'importance': 0.23, 'value': 0.91},
            {'name': 'Feature_D', 'importance': 0.15, 'value': 0.43}
        ],
        'probabilities': [
            {'class': 'B', 'probability': 0.88},
            {'class': 'A', 'probability': 0.08},
            {'class': 'C', 'probability': 0.03},
            {'class': 'D', 'probability': 0.01}
        ]
    }

# --- Dash App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                suppress_callback_exceptions=True)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .svg-viewer-container {
                user-select: none;
                -webkit-user-select: none;
                -moz-user-select: none;
                -ms-user-select: none;
            }
            .svg-viewer-container img {
                pointer-events: none;
            }
        </style>
    </head>
    <body class="bg-slate-50">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
            <script>
                let currentScale = 0.80;
                let currentX = 0;
                let currentY = 0;
                let isDragging = false;
                let startX = 0;
                let startY = 0;
                let startTransformX = 0;
                let startTransformY = 0;

                function updateTransform() {
                    const svgContent = document.getElementById('svg-content');
                    if (svgContent) {
                        svgContent.style.transform = `translate(${currentX}px, ${currentY}px) scale(${currentScale})`;
                        const indicator = document.getElementById('zoom-indicator');
                        if (indicator) {
                            indicator.textContent = Math.round(currentScale * 100) + '%';
                        }
                    }
                }

                function initializeSVGViewer() {
                    const container = document.getElementById('svg-container');
                    const svgContent = document.getElementById('svg-content');
                    if (!container || !svgContent) return;

                    container.addEventListener('wheel', function(e) {
                        e.preventDefault();
                        const delta = e.deltaY > 0 ? 0.9 : 1.1;
                        const newScale = Math.max(0.1, Math.min(5, currentScale * delta));
                        if (newScale !== currentScale) {
                            const rect = container.getBoundingClientRect();
                            const mouseX = e.clientX - rect.left;
                            const mouseY = e.clientY - rect.top;
                            const scaleRatio = newScale / currentScale;
                            currentX = mouseX - (mouseX - currentX) * scaleRatio;
                            currentY = mouseY - (mouseY - currentY) * scaleRatio;
                            currentScale = newScale;
                            updateTransform();
                        }
                    });

                    container.addEventListener('mousedown', function(e) {
                        if (e.button === 0) {
                            isDragging = true;
                            startX = e.clientX;
                            startY = e.clientY;
                            startTransformX = currentX;
                            startTransformY = currentY;
                            container.style.cursor = 'grabbing';
                        }
                    });

                    document.addEventListener('mousemove', function(e) {
                        if (isDragging) {
                            currentX = startTransformX + (e.clientX - startX);
                            currentY = startTransformY + (e.clientY - startY);
                            updateTransform();
                        }
                    });

                    document.addEventListener('mouseup', function() {
                        if (isDragging) {
                            isDragging = false;
                            container.style.cursor = 'move';
                        }
                    });

                    container.addEventListener('contextmenu', e => e.preventDefault());
                    container.style.cursor = 'move';

                    updateTransform();
                }

                // Poll until SVG content exists
                let retries = 0;
                const interval = setInterval(() => {
                    const content = document.getElementById("svg-content");
                    if (content) {
                        initializeSVGViewer();
                        clearInterval(interval);
                    } else if (++retries > 30) {
                        console.warn("⚠️ SVG content not found after waiting.");
                        clearInterval(interval);
                    }
                }, 200);
            </script>
        </footer>
    </body>
</html>
'''

# --- Reusable Components & Layouts ---

def create_header():
    """Creates the main application header and navigation."""
    # This layout matches the structure from Header.tsx and App.tsx
    return html.Header(
        className="bg-white/80 backdrop-blur-md shadow-sm sticky top-0 z-50",
        children=html.Nav(
            className="container mx-auto px-4 flex items-center justify-between h-20",
            children=[
                dcc.Link(href="/", className="flex items-center gap-3", children=[
                    html.I(className="fas fa-rocket text-blue-600 text-3xl"),
                    html.Span("NASA Capstone", className="text-xl font-bold text-gray-800")
                ]),
                html.Div(id="nav-links", className="flex items-center")
            ]
        )
    )

def create_nav_link(label, href, icon_class, active_path):
    """Helper to create a navigation link."""
    is_active = active_path == href
    active_classes = "bg-blue-100 text-blue-700"
    inactive_classes = "text-gray-600 hover:text-gray-900 hover:bg-gray-100"

    return dcc.Link(
        href=href,
        className=f"flex items-center px-3 py-2 rounded-lg transition-colors font-medium {active_classes if is_active else inactive_classes}",
        children=[
            html.I(className=f"{icon_class} w-4 h-4 mr-2"),
            label
        ]
    )

# ... (Home and View Model pages remain largely the same)
def create_page_home():
    """Creates the layout for the home page."""
    return html.Div(
        className="max-w-4xl mx-auto",
        children=[
            html.Div(className="text-center mb-12", children=[
                html.Div(className="flex justify-center mb-6", children=[
                    html.Div(html.I(className="fas fa-rocket fa-3x text-blue-600"), className="p-4 bg-blue-100 rounded-full")
                ]),
                html.H1("Capstone Visualization", className="text-4xl font-bold text-gray-900 mb-4"),
                html.P("Explore and utilize the trained machine learning models with interactive 3D visualizations and classify new data with real-time feature engineering insights.",
                       className="text-xl text-gray-600 max-w-2xl mx-auto")
            ]),
            html.Div(className="grid md:grid-cols-2 gap-8", children=[
                dcc.Link(href="/view-model", className="group cursor-pointer bg-white rounded-xl shadow-lg hover:shadow-xl transition-all p-8 border hover:border-blue-200 block", children=[
                    html.Div(className="flex items-center mb-6", children=[
                        html.Div(html.I(className="fas fa-eye fa-2x text-green-600"), className="p-3 bg-green-100 rounded-lg"),
                        html.H2("View Current Model", className="text-2xl font-semibold text-gray-900 ml-4")
                    ]),
                    html.P("Explore the trained models with interactive decision tree visualization and 3D clustering results. Rotate, zoom, and analyze your data patterns.", className="text-gray-600 mb-6"),
                    html.Div([
                            html.Span("Explore Model", className="text-success fw-medium"),
                            html.I(className="fas fa-bolt ms-2 text-success")
                        ], className="d-flex align-items-center"),
                ]),
                dcc.Link(href="/classify", className="group cursor-pointer bg-white rounded-xl shadow-lg hover:shadow-xl transition-all p-8 border hover:border-purple-200 block", children=[
                    html.Div(className="flex items-center mb-6", children=[
                        html.Div(html.I(className="fas fa-upload fa-2x text-purple-600"), className="p-3 bg-purple-100 rounded-lg"),
                        html.H2("Classify New Run", className="text-2xl font-semibold text-gray-900 ml-4")
                    ]),
                    html.P("Upload CSV data. This will run the feature engineering process and get classification results from the decision tree and the run highlighted in the clustering space.", className="text-gray-600 mb-6"),
                    html.Div([
                            html.Span("Start Classification", className="fw-medium", style={"color": "#7c3aed"}),
                            html.I(className="fas fa-bolt ms-2", style={"color": "#7c3aed"})
                        ], className="d-flex align-items-center"),
                ])
            ])
        ]
    )

def create_page_view_model():
    """Creates the layout for the model viewing page."""
    return html.Div(className="max-w-[90rem] mx-auto px-4", children=[
        html.Div(className="mb-8", children=[
            html.H1("Current Model Overview", className="text-3xl font-bold text-gray-900 mb-2"),
            html.P("Explore the trained models with interactive visualizations.", className="text-gray-600")
        ]),
        html.Div(className="space-y-12", children=[
            # Controls for clustering visualization
            html.Div(className="bg-white rounded-xl shadow-lg p-6 border", children=[
                html.Div(className="flex items-center justify-between mb-6", children=[
                    html.H2("3D Clustering Visualization", className="text-xl font-semibold text-gray-900"),
                    html.Div(className="flex items-center space-x-4", children=[
                        # X-Axis Dropdown
                        html.Div(className="flex items-center space-x-2", children=[
                            html.Label("X-Axis:", className="text-sm font-medium text-gray-700"),
                            dcc.Dropdown(
                                id="x-axis",
                                options=[{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, 7)],
                                value='PCA1',
                                clearable=False,
                                className="w-24"
                            )
                        ]),
                        # Y-Axis Dropdown
                        html.Div(className="flex items-center space-x-2", children=[
                            html.Label("Y-Axis:", className="text-sm font-medium text-gray-700"),
                            dcc.Dropdown(
                                id="y-axis",
                                options=[{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, 7)],
                                value='PCA2',
                                clearable=False,
                                className="w-24"
                            )
                        ]),
                        # Z-Axis Dropdown
                        html.Div(className="flex items-center space-x-2", children=[
                            html.Label("Z-Axis:", className="text-sm font-medium text-gray-700"),
                            dcc.Dropdown(
                                id="z-axis",
                                options=[{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, 7)],
                                value='PCA3',
                                clearable=False,
                                className="w-24"
                            )
                        ]),
                        # Clusters Dropdown
                        html.Div(className="flex items-center space-x-2", children=[
                            html.Label("Clusters:", className="text-sm font-medium text-gray-700"),
                            dcc.Dropdown(
                                id="clusters-count",
                                options=[{'label': str(n), 'value': n} for n in range(2, 20)],
                                value=2,
                                clearable=False,
                                className="w-24"
                            )
                        ])
                    ])
                ]),
                dcc.Graph(id='cluster-plot', style={'height': '500px'})
            ]),
            # Decision Tree Visualization
            html.Div(className="bg-white rounded-xl shadow-lg p-6 border", children=[
                html.H2("Decision Tree", className="text-xl font-semibold text-gray-900 mb-4"),
                create_interactive_svg_viewer()
            ]),
        ])
    ])

# --- CLASSIFICATION PAGE LAYOUTS ---
def create_page_classify():
    """Initial layout for the classification page."""
    return html.Div(className="max-w-6xl mx-auto", children=[
        html.Div(className="mb-8", children=[
            html.H1("Classify New Run", className="text-3xl font-bold text-gray-900 mb-2"),
            html.P("Upload your CSV data to get real-time classification.", className="text-gray-600")
        ]),
        html.Div(id="classify-content-area", children=create_classify_upload_view())
    ])

def create_classify_upload_view(filename=None, filesize=None):
    """View for file upload, matching FileUpload.tsx."""
    if filename:
        return html.Div(className="max-w-2xl mx-auto", children=[
            html.Div(className="bg-white rounded-xl shadow-lg p-8 border border-gray-100", children=[
                html.Div(className="relative border-2 border-dashed rounded-lg p-8 text-center border-green-400 bg-green-50", children=[
                    html.Div(className="flex items-center justify-center space-x-4", children=[
                        html.I(className="fas fa-file-csv fa-2x text-green-600"),
                        html.Div(className="flex-1 text-left", children=[
                            html.P(filename, className="font-medium text-gray-900"),
                            html.P(f"{filesize:.1f} KB", className="text-sm text-gray-500")
                        ]),
                        html.Button(html.I(className="fas fa-times text-red-500"), id="remove-file-btn", className="p-1 hover:bg-red-100 rounded-full")
                    ])
                ]),
                html.Div(className="mt-6 flex justify-center", children=[
                    html.Button(id="start-classify-btn", n_clicks=0, className="px-8 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-medium flex items-center space-x-2", children=[
                        html.I(className="fas fa-play w-5 h-5"),
                        html.Span("Start Classification")
                    ])
                ])
            ])
        ])

    return html.Div(className="max-w-2xl mx-auto", children=[
        html.Div(className="bg-white rounded-xl shadow-lg p-8 border border-gray-100", children=[
            dcc.Upload(
                id='upload-data',
                className="!border-dashed !border-2 !border-gray-300 !rounded-xl !p-12 text-center cursor-pointer hover:!border-blue-500 !bg-white/50 transition-colors",
                children=html.Div([
                    html.I(className="fas fa-upload w-12 h-12 text-gray-400 mx-auto mb-4"),
                    html.P("Drop your CSV file here, or click to browse", className="text-lg font-medium text-gray-900 mb-2")
                ])
            )
        ])
    ])

def create_classify_processing_view(filename):
    """View for processing, matching ProcessingSteps.tsx."""
    steps = [
        {'name': 'Feature Extraction', 'desc': 'Extracting numerical features'},
        {'name': 'Model Prediction', 'desc': 'Running classification algorithm'},
        {'name': 'Result Generation', 'desc': 'Preparing visualization and results'}
    ]
    return html.Div(className="max-w-3xl mx-auto", children=[
        html.Div(className="bg-white rounded-xl shadow-lg p-8 border", children=[
            html.Div(className="text-center mb-8", children=[
                html.H2("Processing Your Data", className="text-2xl font-semibold text-gray-900 mb-2"),
                html.P(f"Analyzing {filename}...", className="text-gray-600"),
                dbc.Spinner(color="primary", spinner_class_name="mt-4"),
            ]),
            html.Div(id="processing-steps-container", className="space-y-6", children=[
                # Steps will be populated by the interval callback
            ])
        ])
    ])

def create_classify_results_view(result_data):
    """View for results, matching ClassificationResult.tsx and the provided screenshot, with clustering controls and consistent layout."""
    if not result_data: return html.Div("Error: No result data found.")

    res = result_data['result']
    analysis = res['analysis']
    predicted_class = res['class']
    regime_name = REGIME_MAPPING.get(predicted_class, "Unknown")

    # --- Success Message ---
    success_message = html.Div(
        className="flex flex-col items-center justify-center bg-green-50 rounded-xl py-8 mb-8 border border-green-200 w-full",
        children=[
            html.H2("Classification Complete!", className="text-2xl font-bold text-green-700 mb-2"),
            html.Div(
                className="flex items-center justify-center mb-2",
                children=[
                    html.Div(
                        str(predicted_class + 1),
                        className="rounded-full bg-green-200 text-green-800 text-3xl font-bold w-16 h-16 flex items-center justify-center shadow"
                    )
                ]
            ),
            html.Div([
                html.Span("Predicted Class: ", className="font-semibold text-gray-700"),
                html.Span(f"{predicted_class + 1} - {regime_name}", className="text-lg font-bold text-green-700")
            ], className="mb-2"),
            html.P("Your data point has been classified and highlighted in the 3D visualization below", className="text-gray-600 mb-4"),
            html.Button(
                "Classify Another Run",
                id="reset-classify-btn",
                className="px-6 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 font-medium"
            )
        ]
    )

    # --- Decision Tree Visualization (same as view model) ---
    tree_panel = html.Div(
        className="bg-white rounded-xl shadow-lg p-6 border mb-12 w-full",
        children=[
            html.H2("Decision Tree", className="text-xl font-semibold text-gray-900 mb-4"),
            create_interactive_svg_viewer()
        ]
    )

    # --- Clustering Controls ---
    controls = html.Div(
        className="flex items-center space-x-4 mb-4 w-full",
        children=[
            html.Div(className="flex items-center space-x-2", children=[
                html.Label("X-Axis:", className="text-sm font-medium text-gray-700"),
                dcc.Dropdown(
                    id="classify-x-axis",
                    options=[{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, 7)],
                    value='PCA1',
                    clearable=False,
                    className="w-24"
                )
            ]),
            html.Div(className="flex items-center space-x-2", children=[
                html.Label("Y-Axis:", className="text-sm font-medium text-gray-700"),
                dcc.Dropdown(
                    id="classify-y-axis",
                    options=[{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, 7)],
                    value='PCA2',
                    clearable=False,
                    className="w-24"
                )
            ]),
            html.Div(className="flex items-center space-x-2", children=[
                html.Label("Z-Axis:", className="text-sm font-medium text-gray-700"),
                dcc.Dropdown(
                    id="classify-z-axis",
                    options=[{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, 7)],
                    value='PCA3',
                    clearable=False,
                    className="w-24"
                )
            ]),
            html.Div(className="flex items-center space-x-2", children=[
                html.Label("Clusters:", className="text-sm font-medium text-gray-700"),
                dcc.Dropdown(
                    id="classify-clusters-count",
                    options=[{'label': str(n), 'value': n} for n in range(2, 20)],
                    value=2,
                    clearable=False,
                    className="w-24"
                )
            ])
        ]
    )

    # --- 3D Clustering with New Point ---
    initial_figure = get_3d_cluster_data(
        new_point=res['point'],
        x_axis='PCA1',
        y_axis='PCA2',
        z_axis='PCA3',
        clusters_count=2
    )
    cluster_panel = html.Div(
        className="bg-white rounded-xl shadow-lg p-6 border flex flex-col items-center mb-12 w-full",
        children=[
            html.H3("3D Clustering with New Point", className="text-xl font-semibold text-gray-900 mb-4 w-full text-left"),
            controls,
            dcc.Graph(id='classify-cluster-plot', figure=initial_figure, style={'height': '350px', 'width': '100%'}),
            html.Div(
                className="mt-4 w-full",
                children=html.Div(
                    className="flex items-center bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded",
                    children=[
                        html.I(className="fas fa-circle text-yellow-400 mr-3"),
                        html.Div([
                            html.Span("New Data Point", className="font-semibold text-yellow-700"),
                            html.Br(),
                            html.Span(f"Your classified data point is highlighted.", className="text-yellow-700")
                        ])
                    ]
                )
            )
        ]
    )

    # --- Layout ---
    return html.Div(
        className="max-w-[90rem] mx-auto px-4 py-8",
        children=[
            success_message,
            tree_panel,
            cluster_panel,
            dcc.Store(id='classified-point-store', data=res['point'])
        ]
    )

# --- Callback for classify cluster plot ---
@app.callback(
    Output('classify-cluster-plot', 'figure'),
    Input('classify-x-axis', 'value'),
    Input('classify-y-axis', 'value'),
    Input('classify-z-axis', 'value'),
    Input('classify-clusters-count', 'value'),
    State('classified-point-store', 'data'),
    prevent_initial_call=True
)
def update_classify_cluster_plot(x_axis, y_axis, z_axis, clusters_count, classified_point):
    """Updates the cluster plot on the classification results page based on selected options."""
    return get_3d_cluster_data(
        new_point=classified_point,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        clusters_count=clusters_count
    )

# --- Main App Layout ---
app.layout = html.Div(
    className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 font-sans",
    children=[
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='uploaded-file-store'),
        dcc.Store(id='classification-result-store'),
        dcc.Store(id='svg-transform-store', data={'scale': 0.80, 'x': 0, 'y': 0}),
        dcc.Interval(id='processing-interval', interval=800, n_intervals=0, disabled=True),
        create_header(),
        html.Main(id='page-content', className="container mx-auto px-4 py-8")
    ]
)

# --- Callbacks ---

@app.callback(
    Output('page-content', 'children'),
    Output('nav-links', 'children'),
    Input('url', 'pathname')
)
def router_and_nav(pathname):
    """Router to display pages and update nav link styles."""
    nav_links = [
        create_nav_link("View Model", "/view-model", "fas fa-eye", pathname),
        create_nav_link("Classify", "/classify", "fas fa-upload", pathname),
    ]
    if pathname == '/view-model':
        return create_page_view_model(), nav_links
    elif pathname == '/classify':
        return create_page_classify(), nav_links
    else:
        return create_page_home(), nav_links

# SVG Control Callbacks
@app.callback(
    Output('svg-transform-store', 'data'),
    [Input('zoom-in-btn', 'n_clicks'),
     Input('zoom-out-btn', 'n_clicks'),
     Input('reset-view-btn', 'n_clicks')],
    State('svg-transform-store', 'data'),
    prevent_initial_call=True
)
def handle_svg_controls(zoom_in_clicks, zoom_out_clicks, reset_clicks, current_transform):
    """Handle SVG zoom and reset controls."""
    ctx = callback_context
    if not ctx.triggered:
        return current_transform

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'zoom-in-btn':
        new_scale = min(5, current_transform['scale'] * 1.2)
        return {**current_transform, 'scale': new_scale}
    elif button_id == 'zoom-out-btn':
        new_scale = max(0.1, current_transform['scale'] * 0.8)
        return {**current_transform, 'scale': new_scale}
    elif button_id == 'reset-view-btn':
        return {'scale': 0.80, 'x': 0, 'y': 0}

    return current_transform

@app.callback(
    Output('uploaded-file-store', 'data'),
    Output('classify-content-area', 'children'),
    Input('upload-data', 'contents'),
   # Input('remove-file-btn', 'n_clicks'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    prevent_initial_call=True
)
def handle_file_upload(contents, filename, last_modified):
    """Handles storing uploaded file info and updating the upload UI."""
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    # Safe check for remove-file-btn via callback_context
    if triggered_id == 'remove-file-btn':
        return None, create_classify_upload_view()

    if contents:
        content_type, content_string = contents.split(',')

        try:
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            file_size_kb = len(decoded) / 1024

            file_data = {
                'filename': filename,
                'filesize': file_size_kb,
                'contents': contents,
                'data': df.to_json(date_format='iso', orient='split')  # ✅ required for classification
            }

            return file_data, create_classify_upload_view(filename=filename, filesize=file_size_kb)

        except Exception as e:
            return None, html.Div(className="text-red-600 text-center p-4", children=[
                html.I(className="fas fa-exclamation-triangle mr-2"),
                f"Error reading file: {str(e)}"
            ])

    return None, create_classify_upload_view()

@app.callback(
    Output('classify-content-area', 'children', allow_duplicate=True),
    Output('processing-interval', 'disabled'),
    Output('classification-result-store', 'data'),
    Input('start-classify-btn', 'n_clicks'),
    State('uploaded-file-store', 'data'),
    prevent_initial_call=True
)
def start_classification(n_clicks, file_data):
    """Starts the ML pipeline when classification is requested."""
    if not n_clicks or not file_data:
        return dash.no_update, True, None

    filename = file_data['filename']
    df = pd.read_json(io.StringIO(file_data['data']), orient='split')

    # Start processing view
    return create_classify_processing_view(filename), False, None

@app.callback(
    Output('processing-steps-container', 'children'),
    Input('processing-interval', 'n_intervals'),
    State('processing-interval', 'disabled'),
    prevent_initial_call=True
)
def update_processing_steps(n_intervals, is_disabled):
    """Updates the processing steps animation."""
    if is_disabled:
        return dash.no_update

    steps = [
        {'name': 'Feature Extraction', 'desc': 'Extracting numerical features'},
        {'name': 'Model Prediction', 'desc': 'Running classification algorithm'},
        {'name': 'Result Generation', 'desc': 'Preparing visualization and results'}
    ]

    current_step = min(n_intervals // 2, len(steps) - 1)  # Each step takes ~1.6 seconds

    step_elements = []
    for i, step in enumerate(steps):
        if i < current_step:
            # Completed step
            step_elem = html.Div(className="flex items-center space-x-4 p-4 bg-green-50 border border-green-200 rounded-lg", children=[
                html.I(className="fas fa-check-circle text-green-600 text-xl"),
                html.Div([
                    html.H4(step['name'], className="font-medium text-green-900"),
                    html.P(step['desc'], className="text-sm text-green-700")
                ])
            ])
        elif i == current_step:
            # Current step
            step_elem = html.Div(className="flex items-center space-x-4 p-4 bg-blue-50 border border-blue-200 rounded-lg", children=[
                dbc.Spinner(color="primary", size="sm"),
                html.Div([
                    html.H4(step['name'], className="font-medium text-blue-900"),
                    html.P(step['desc'], className="text-sm text-blue-700")
                ])
            ])
        else:
            # Pending step
            step_elem = html.Div(className="flex items-center space-x-4 p-4 bg-gray-50 border border-gray-200 rounded-lg", children=[
                html.I(className="fas fa-circle text-gray-400 text-xl"),
                html.Div([
                    html.H4(step['name'], className="font-medium text-gray-600"),
                    html.P(step['desc'], className="text-sm text-gray-500")
                ])
            ])
        step_elements.append(step_elem)

    return step_elements

@app.callback(
    Output('classify-content-area', 'children', allow_duplicate=True),
    Output('processing-interval', 'disabled', allow_duplicate=True),
    Output('classification-result-store', 'data', allow_duplicate=True),
    Input('processing-interval', 'n_intervals'),
    State('uploaded-file-store', 'data'),
    State('processing-interval', 'disabled'),
    prevent_initial_call=True
)
def complete_classification(n_intervals, file_data, is_disabled):
    """Completes classification after processing steps."""
    if is_disabled or not file_data:
        return dash.no_update, dash.no_update, dash.no_update

    # Complete after 5 steps (about 5 seconds)
    if n_intervals >= 10:  # 5 steps * 2 intervals per step
        df = pd.read_json(io.StringIO(file_data['data']), orient='split')
        result = run_ml_pipeline(df)

        result_data = {
            'result': result,
            'filename': file_data['filename'],
            'timestamp': time.time()
        }

        return create_classify_results_view(result_data), True, result_data

    return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('classify-content-area', 'children', allow_duplicate=True),
    Output('uploaded-file-store', 'data', allow_duplicate=True),
    Input('reset-classify-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_classification(n_clicks):
    """Resets the classification process to start over."""
    if not n_clicks:
        return dash.no_update, dash.no_update

    return create_classify_upload_view(), None

@app.callback(
    Output('classify-content-area', 'children', allow_duplicate=True),
    Output('uploaded-file-store', 'data', allow_duplicate=True),
    Input('remove-file-btn', 'n_clicks'),
    prevent_initial_call=True
)
def remove_uploaded_file(n_clicks):
    """Removes the uploaded file and returns to upload view."""
    if not n_clicks:
        return dash.no_update, dash.no_update

    return create_classify_upload_view(), None

# Download SVG callback (optional - requires server-side file handling)
@app.callback(
    Output('download-svg-btn', 'n_clicks'),
    Input('download-svg-btn', 'n_clicks'),
    prevent_initial_call=True
)
def download_svg(n_clicks):
    """Handles SVG download - would need server-side implementation."""
    if n_clicks:
        # In a real implementation, you'd trigger a download here
        # For now, just reset the click count
        print("SVG download requested")
    return 0

app.clientside_callback(
    """
    function(transform) {
        const svgContent = document.getElementById('svg-content');
        const indicator = document.getElementById('zoom-indicator');
        if (svgContent) {
            svgContent.style.transform = `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`;
        }
        if (indicator) {
            indicator.textContent = Math.round(transform.scale * 100) + '%';
        }
        return '';
    }
    """,
    Output('zoom-indicator', 'children'),
    Input('svg-transform-store', 'data')
)

# Update callback for the cluster plot
@app.callback(
    Output('cluster-plot', 'figure'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    Input('z-axis', 'value'),
    Input('clusters-count', 'value')
)
def update_cluster_plot(x_axis, y_axis, z_axis, clusters_count):
    """Updates the cluster plot based on selected options."""
    return get_3d_cluster_data(
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        clusters_count=clusters_count
    )

server = app.server

if __name__ == '__main__':
    app.run(debug=False)