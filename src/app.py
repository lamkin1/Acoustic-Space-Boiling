import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import os
from pathlib import Path

def create_dash_app(df_dict, directory):
    """
    Parameters:
      df_dict: Dictionary with keys (2,3,...,6) corresponding to dataframes
               that contain 'PCA1', 'PCA2', ... plus 'Cluster' and 'file_name'
      directory: Base directory (used to locate PNG files)
    """
    app = dash.Dash(__name__)

    # Define dropdown options for number of PCA components (2 to 6)
    pca_count_options = [{'label': str(n), 'value': n} for n in range(2, 7)]
    default_pca_count = 2

    # Initial axis options for PCA components (for 2 components)
    axis_options = [{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, default_pca_count + 1)]

    # Create an initial figure using the dataframe for 2 components (2D scatter)
    dff = df_dict[default_pca_count]
    default_fig = px.scatter(
        dff,
        x="PCA1",
        y="PCA2",
        color=dff["Cluster"].astype(str),
        hover_data=["file_name"],
        category_orders={"Cluster": sorted(dff["Cluster"].unique())},
        labels={"color": "Cluster"}
    )
    default_fig.update_traces(
        marker=dict(size=8),
        customdata=dff["file_name"],
        hovertemplate=None
    )
    default_fig.update_layout(dragmode="select")

    app.layout = html.Div([
        html.Div([
            html.Label("Number of PCA Components:"),
            dcc.Dropdown(
                id="pca-count",
                options=pca_count_options,
                value=default_pca_count
            )
        ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
        html.Div([
            html.Label("X-Axis:"),
            dcc.Dropdown(id="xaxis", options=axis_options, value="PCA1")
        ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
        html.Div([
            html.Label("Y-Axis:"),
            dcc.Dropdown(id="yaxis", options=axis_options, value="PCA2")
        ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
        # The Z-axis dropdown is conditionally shown when more than 2 components are available.
        html.Div([
            html.Label("Z-Axis:"),
            dcc.Dropdown(id="zaxis", options=[], value="PCA3")
        ], id="zaxis-div", style={"display": "none", "width": "30%", "display": "inline-block", "padding": "10px"}),
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
        # Only provide options for Z if more than 2 components are selected.
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

    # Update the scatter plot based on selected axes and PCA count.
    @app.callback(
        Output("scatter-plot", "figure"),
        Output("current-yaxis", "data"),
        Input("xaxis", "value"),
        Input("yaxis", "value"),
        Input("zaxis", "value"),
        Input("pca-count", "value")
    )
    def update_scatter(xaxis, yaxis, zaxis, pca_count):
        dff = df_dict[pca_count]
        if pca_count > 2:
            # Create a 3D scatter plot if more than 2 components are available.
            fig = px.scatter_3d(
                dff,
                x=xaxis,
                y=yaxis,
                z=zaxis,
                color=dff["Cluster"].astype(str),
                hover_data=["file_name"],
                category_orders={"Cluster": sorted(dff["Cluster"].unique())},
                labels={"color": "Cluster"}
            )
        else:
            # Create a 2D scatter plot for 2 components.
            fig = px.scatter(
                dff,
                x=xaxis,
                y=yaxis,
                color=dff["Cluster"].astype(str),
                hover_data=["file_name"],
                category_orders={"Cluster": sorted(dff["Cluster"].unique())},
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
        # Construct the full file system path for checking file existence.
        image_path = os.path.join(directory, "assets", png_file)
        if not os.path.exists(image_path):
            return False, dash.no_update, dash.no_update, dash.no_update

        children = html.Div([
            html.Img(src=f"/assets/{png_file}", style={"width": "200px"}),
        ])
        y = hover_data["y"]
        # (Using a dummy median value for tooltip direction here.)
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

        # Keep only the last two selected images.
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
        print(selected_files)

        img1_src = f"/assets/{selected_files[0]}.png" if len(selected_files) > 0 else ""
        img2_src = f"/assets/{selected_files[1]}.png" if len(selected_files) > 1 else ""
        print(img1_src)

        img1_style = {"width": "100%", "display": "block"} if img1_src else {"display": "none"}
        img2_style = {"width": "100%", "display": "block"} if img2_src else {"display": "none"}

        return img1_src, img1_style, img2_src, img2_style

    return app

# --- Initialize the app ---

from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parents[0]

print(project_root)

# Load different CSV files for each PCA component count.
df_2 = pd.read_csv(project_root / "/assets/pcaDF_2.csv")
df_3 = pd.read_csv(project_root / "/assets/pcaDF_3.csv")
df_4 = pd.read_csv(project_root / "/assets/pcaDF_4.csv")
df_5 = pd.read_csv(project_root / "/assets/pcaDF_5.csv")
df_6 = pd.read_csv(project_root / "/assets/pcaDF_6.csv")

# Optionally remove unwanted prefix from file_name column.
for df in [df_2, df_3, df_4, df_5, df_6]:
    df["file_name"] = df["file_name"].str[15:]

# Store the dataframes in a dictionary keyed by the number of PCA components.
df_dict = {
    2: df_2,
    3: df_3,
    4: df_4,
    5: df_5,
    6: df_6
}

app = create_dash_app(df_dict, project_root)

# Export the underlying Flask server for production
server = app.server

if __name__ == '__main__':
    app.run(debug=True)
