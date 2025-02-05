import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
import shutil
from pathlib import Path


def create_dash_app(df, directory):
    """
    df: DataFrame containing 'PCA1', 'PCA2', 'Cluster', and 'file_name'.
    directory: Base directory.
    """
    png_source_dir = os.path.join(directory / "Data/pngs")

    png_target_dir = "assets/pngs"
    os.makedirs(png_target_dir, exist_ok=True)

    for file in os.listdir(png_source_dir):
        if file.endswith(".png"):
            shutil.copy(os.path.join(png_source_dir, file), os.path.join(png_target_dir, file))

    app = dash.Dash(__name__)

    fig = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        color=df["Cluster"].astype(str),  # Keep Cluster discrete
        hover_data=["file_name"],
        category_orders={"Cluster": sorted(df["Cluster"].unique())},  # Maintain integer order
        labels={"color": "Cluster"}  # Rename legend label
    )

    fig.update_traces(
        marker=dict(size=8),
        customdata=df["file_name"],
        hovertemplate=None
    )

    app.layout = html.Div([
        dcc.Graph(id='scatter-plot', figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip", direction="bottom"),
    ])

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Output("graph-tooltip", "direction"),
        Input("scatter-plot", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, dash.no_update, dash.no_update, dash.no_update

        # Get file name from hover event
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        file_name = hover_data["customdata"]

        # Convert CSV filename to PNG filename
        png_file = file_name + ".png"
        image_path = f"assets/pngs/{png_file}"  # Path inside assets

        if not os.path.exists(image_path):
            return False, dash.no_update, dash.no_update, dash.no_update

        # Construct the tooltip with the image
        children = html.Div([
            html.Img(src=f"/{image_path}", style={"width": "200px"}),
        ])

        # Tooltip position logic
        y = hover_data["y"]
        direction = "bottom" if y > df["PCA2"].median() else "top"

        return True, bbox, children, direction

    return app


script_path = Path(__file__).resolve()
project_root = script_path.parents[0]

df = pd.read_csv(project_root / "pcaDF.csv")
app = create_dash_app(df, project_root)
app.run(debug=True)