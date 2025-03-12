import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import os
from pathlib import Path

def create_dash_app(results_dict, directory):
    """
    Parameters:
      results_dict: Dictionary with keys (2,3,...,7) corresponding to precomputed clustering results CSVs.
                    Each DataFrame contains columns: 'PCA1', 'PCA2', ... 'PCA6', 'Cluster_2_PCs', ..., 'Cluster_6_PCs',
                    'file_name' and 'date'.
      directory: Base directory (used to locate PNG files)
    """
    app = dash.Dash(__name__)

    # Dropdown options for number of PCA components (2 to 6)
    pca_count_options = [{'label': str(n), 'value': n} for n in range(2, 7)]
    default_pca_count = 2

    # Dropdown options for number of clusters (results files exist for 2 to 7 clusters)
    clusters_options = [{'label': str(n), 'value': n} for n in range(2, 8)]
    default_clusters = 2

    # Initial axis options (for 2 PCA components)
    axis_options = [{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, default_pca_count + 1)]

    # Load the default DataFrame from the results file for default_clusters
    dff = results_dict[default_clusters]
    # Use the cluster column corresponding to the default number of PCA components.
    cluster_col = f'Cluster_{default_pca_count}_PCs'

    # Create an initial 2D scatter plot using PCA1 and PCA2
    default_fig = px.scatter(
        dff,
        x="PCA1",
        y="PCA2",
        color=dff[cluster_col].astype(str),
        hover_data=["file_name"],
        category_orders={cluster_col: sorted(dff[cluster_col].unique())},
        labels={"color": "Cluster"}
    )
    default_fig.update_traces(
        marker=dict(size=8),
        customdata=dff["file_name"],
        hovertemplate=None
    )
    default_fig.update_layout(dragmode="select")

    # Main app layout, including both dropdowns for PCA components and clusters.
    app.layout = html.Div([
        # First row: PCA and Clusters dropdowns
        html.Div([
            html.Div([
                html.Label("Number of PCA Components:"),
                dcc.Dropdown(
                    id="pca-count",
                    options=pca_count_options,
                    value=default_pca_count
                )
            ], style={"width": "45%", "padding": "10px"}),
            html.Div([
                html.Label("Number of Clusters:"),
                dcc.Dropdown(
                    id="clusters-count",
                    options=clusters_options,
                    value=default_clusters
                )
            ], style={"width": "45%", "padding": "10px"})
        ], style={"display": "flex", "justify-content": "space-around"}),

        # Second row: Axis dropdowns
        html.Div([
            html.Div([
                html.Label("X-Axis:"),
                dcc.Dropdown(id="xaxis", options=axis_options, value="PCA1")
            ], style={"width": "30%", "padding": "10px"}),
            html.Div([
                html.Label("Y-Axis:"),
                dcc.Dropdown(id="yaxis", options=axis_options, value="PCA2")
            ], style={"width": "30%", "padding": "10px"}),
            html.Div([
                html.Label("Z-Axis:"),
                dcc.Dropdown(id="zaxis", options=[], value="PCA3")
            ], id="zaxis-div", style={"display": "none", "width": "30%", "padding": "10px"})
        ], style={"display": "flex", "justify-content": "space-around"}),

        dcc.Store(id="current-yaxis", data="PCA2"),
        dcc.Graph(id='scatter-plot', figure=default_fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip", direction="bottom"),
        dcc.Store(id="selected-points", data=[]),
        html.Div(id="comparison-section", children=[
            html.H3("Comparison View", style={"text-align": "center", "margin-top": "20px"}),
            html.Div([
                html.Div([html.Img(id="img1", style={"width": "100%", "display": "none"})],
                         style={"text-align": "center", "width": "40%", "padding": "10px"}),
                html.Div([html.Img(id="img2", style={"width": "100%", "display": "none"})],
                         style={"text-align": "center", "width": "40%", "padding": "10px"})
            ], style={"display": "flex", "justify-content": "center", "align-items": "center"})
        ])
    ])

    # --- Callbacks ---

    # Update dropdown options for X, Y, and Z axes based on the number of PCA components selected.
    @app.callback(
        Output("xaxis", "options"),
        Output("yaxis", "options"),
        Output("zaxis", "options"),
        Input("pca-count", "value")
    )
    def update_axis_options(pca_count):
        options = [{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, pca_count + 1)]
        z_options = options if pca_count > 2 else []
        return options, options, z_options

    # Toggle visibility of the Z-axis dropdown div based on PCA count.
    @app.callback(
        Output("zaxis-div", "style"),
        Input("pca-count", "value")
    )
    def toggle_zaxis_div(pca_count):
        if pca_count > 2:
            return {"width": "30%", "display": "inline-block", "padding": "10px"}
        else:
            return {"display": "none"}

    # Update the scatter plot based on selected axes, PCA components, and clusters.
    @app.callback(
        Output("scatter-plot", "figure"),
        Output("current-yaxis", "data"),
        Input("xaxis", "value"),
        Input("yaxis", "value"),
        Input("zaxis", "value"),
        Input("pca-count", "value"),
        Input("clusters-count", "value")
    )
    def update_scatter(xaxis, yaxis, zaxis, pca_count, clusters_count):
        # Retrieve the appropriate DataFrame based on the number of clusters.
        dff = results_dict[clusters_count]
        # Determine which cluster column to use based on the selected number of PCA components.
        cluster_col = f'Cluster_{pca_count}_PCs'
        if pca_count > 2:
            fig = px.scatter_3d(
                dff,
                x=xaxis,
                y=yaxis,
                z=zaxis,
                color=dff[cluster_col].astype(str),
                hover_data=["file_name"],
                category_orders={cluster_col: sorted(dff[cluster_col].unique())},
                labels={"color": "Cluster"}
            )
        else:
            fig = px.scatter(
                dff,
                x=xaxis,
                y=yaxis,
                color=dff[cluster_col].astype(str),
                hover_data=["file_name"],
                category_orders={cluster_col: sorted(dff[cluster_col].unique())},
                labels={"color": "Cluster"}
            )
        fig.update_traces(
            marker=dict(size=8),
            customdata=dff["file_name"],
            hovertemplate=None
        )
        fig.update_layout(dragmode="select")
        return fig, yaxis

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Output("graph-tooltip", "direction"),
        Input("scatter-plot", "hoverData"),
        Input("current-yaxis", "data")
    )
    def display_hover(hoverData, current_yaxis):
        if hoverData is None:
            return False, dash.no_update, dash.no_update, dash.no_update

        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        file_name = hover_data["customdata"]
        png_file = file_name + ".png"
        image_path = os.path.join(directory, "assets", png_file)
        if not os.path.exists(image_path):
            return False, dash.no_update, dash.no_update, dash.no_update

        children = html.Div([
            html.Img(src=f"/assets/{png_file}", style={"width": "200px"}),
        ])
        y = hover_data["y"]
        median_val = 0
        direction = "bottom" if y > median_val else "top"

        return True, bbox, children, direction

    @app.callback(
        Output("selected-points", "data"),
        Input("scatter-plot", "clickData"),
        State("selected-points", "data")
    )
    def update_selected_points(clickData, stored_points):
        if clickData is None or "points" not in clickData:
            return stored_points

        file_name = clickData["points"][0]["customdata"]

        if not isinstance(stored_points, list):
            stored_points = []

        if file_name not in stored_points:
            stored_points.append(file_name)

        if len(stored_points) > 2:
            stored_points = stored_points[-2:]

        return stored_points

    @app.callback(
        Output("img1", "src"),
        Output("img1", "style"),
        Output("img2", "src"),
        Output("img2", "style"),
        Input("selected-points", "data")
    )
    def update_images(selected_files):
        if len(selected_files) == 0:
            return "", {"display": "none"}, "", {"display": "none"}
        img1_src = f"/assets/{selected_files[0]}.png" if len(selected_files) > 0 else ""
        img2_src = f"/assets/{selected_files[1]}.png" if len(selected_files) > 1 else ""
        img1_style = {"width": "100%", "display": "block"} if img1_src else {"display": "none"}
        img2_style = {"width": "100%", "display": "block"} if img2_src else {"display": "none"}
        return img1_src, img1_style, img2_src, img2_style

    return app

# --- Initialize the app ---

from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parents[0]

# Load the results CSV files from the assets folder.
results_2 = pd.read_csv(project_root / "assets/results_2_clusters.csv")
results_3 = pd.read_csv(project_root / "assets/results_3_clusters.csv")
results_4 = pd.read_csv(project_root / "assets/results_4_clusters.csv")
results_5 = pd.read_csv(project_root / "assets/results_5_clusters.csv")
results_6 = pd.read_csv(project_root / "assets/results_6_clusters.csv")
results_7 = pd.read_csv(project_root / "assets/results_7_clusters.csv")

# Optionally remove unwanted prefix from file_name column.
for df in [results_2, results_3, results_4, results_5, results_6, results_7]:
    df["file_name"] = df["file_name"].str[15:]

# Create a dictionary keyed by the number of clusters.
results_dict = {
    2: results_2,
    3: results_3,
    4: results_4,
    5: results_5,
    6: results_6,
    7: results_7
}

app = create_dash_app(results_dict, project_root)

# --- Add Simple Password Authentication ---
main_layout = app.layout

login_layout = html.Div([
    html.H2("Please Log In"),
    dcc.Input(id='password-input', type='password', placeholder='Enter Password'),
    html.Button("Submit", id='login-button'),
    html.Div(id='login-output', style={'color': 'red'})
], style={'text-align': 'center', 'margin-top': '100px'})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='logged-in', storage_type='session'),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    Input('logged-in', 'data')
)
def render_page(logged_in):
    if logged_in:
        return main_layout
    return login_layout

@app.callback(
    Output('logged-in', 'data'),
    Output('login-output', 'children'),
    Input('login-button', 'n_clicks'),
    State('password-input', 'value'),
    prevent_initial_call=True
)
def verify_password(n_clicks, password):
    if password == "NASA_Boiling":
        return True, ""
    return dash.no_update, "Incorrect password. Please try again."

# Export the underlying Flask server for production
server = app.server

if __name__ == '__main__':
    app.run(debug=False)
