import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import os
import shutil
from pathlib import Path

def create_dash_app(df, directory):
    """
    Parameters:
      df: DataFrame containing 'PCA1' ... 'PCA6', 'Cluster', and 'file_name'
      directory: Base directory (used to locate png files)
    """
    # Copy png files into the assets folder (so Dash can serve them)
    png_source_dir = os.path.join(directory / "Data/pngs")
    png_target_dir = "assets/pngs"
    os.makedirs(png_target_dir, exist_ok=True)

    for file in os.listdir(png_source_dir):
        if file.endswith(".png"):
            shutil.copy(os.path.join(png_source_dir, file), os.path.join(png_target_dir, file))

    app = dash.Dash(__name__)

    # Define dropdown options for PCA components
    pca_options = [{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, 7)]

    # Create an initial figure (default: PCA1 vs PCA2)
    default_fig = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        color=df["Cluster"].astype(str),
        hover_data=["file_name"],
        category_orders={"Cluster": sorted(df["Cluster"].unique())},
        labels={"color": "Cluster"}
    )
    default_fig.update_traces(
        marker=dict(size=8),
        customdata=df["file_name"],
        hovertemplate=None
    )
    default_fig.update_layout(dragmode="select")

    app.layout = html.Div([
        html.Div([
            html.Label("X-Axis:"),
            dcc.Dropdown(
                id="xaxis",
                options=pca_options,
                value="PCA1"
            )
        ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),

        html.Div([
            html.Label("Y-Axis:"),
            dcc.Dropdown(
                id="yaxis",
                options=pca_options,
                value="PCA2"
            )
        ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),

        # This store will hold the current y-axis column name so that the tooltip callback
        # can compute the median based on the correct column.
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

    # Callback to update the scatter plot (and store the current y-axis) based on dropdown selections.
    @app.callback(
        Output("scatter-plot", "figure"),
        Output("current-yaxis", "data"),
        Input("xaxis", "value"),
        Input("yaxis", "value")
    )
    def update_scatter(xaxis, yaxis):
        fig = px.scatter(
            df,
            x=xaxis,
            y=yaxis,
            color=df["Cluster"].astype(str),
            hover_data=["file_name"],
            category_orders={"Cluster": sorted(df["Cluster"].unique())},
            labels={"color": "Cluster"}
        )
        fig.update_traces(
            marker=dict(size=8),
            customdata=df["file_name"],
            hovertemplate=None
        )
        fig.update_layout(dragmode="select")
        return fig, yaxis

    # Callback to display a tooltip with the image when hovering over a point.
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
        image_path = f"assets/pngs/{png_file}"

        if not os.path.exists(image_path):
            return False, dash.no_update, dash.no_update, dash.no_update

        children = html.Div([
            html.Img(src=f"/{image_path}", style={"width": "200px"}),
        ])

        y = hover_data["y"]
        # Compute the median for the selected y-axis (so tooltip direction makes sense for the displayed plot)
        median_val = df[current_yaxis].median() if current_yaxis in df.columns else 0
        direction = "bottom" if y > median_val else "top"

        return True, bbox, children, direction

    # Callback to update the stored selected points when clicking on the scatter plot.
    @app.callback(
        Output("selected-points", "data"),
        Input("scatter-plot", "clickData"),
        State("selected-points", "data")
    )
    def update_selected_points(clickData, stored_points):
        print("Raw clickData:", clickData)
        if clickData is None or "points" not in clickData:
            print("No point clicked. Returning current selection:", stored_points)
            return stored_points

        file_name = clickData["points"][0]["customdata"]

        if not isinstance(stored_points, list):
            stored_points = []

        if file_name not in stored_points:
            stored_points.append(file_name)

        # Keep only the last two selected images.
        if len(stored_points) > 2:
            stored_points = stored_points[-2:]

        print("Updated selected points:", stored_points)
        return stored_points

    # Callback to update the comparison images based on the selected points.
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

        img1_src = f"/assets/pngs/{selected_files[0]}.png" if len(selected_files) > 0 else ""
        img2_src = f"/assets/pngs/{selected_files[1]}.png" if len(selected_files) > 1 else ""

        img1_style = {"width": "100%", "display": "block"} if img1_src else {"display": "none"}
        img2_style = {"width": "100%", "display": "block"} if img2_src else {"display": "none"}

        return img1_src, img1_style, img2_src, img2_style

    return app

# Determine the project root (adjust as needed)
script_path = Path(__file__).resolve()
project_root = script_path.parents[0]

# Load the CSV file (make sure it has PCA1-PCA6, Cluster, file_name)
df = pd.read_csv(project_root / "Clustering/Scripts/pcaDF.csv")

app = create_dash_app(df, project_root)
app.run(debug=True)