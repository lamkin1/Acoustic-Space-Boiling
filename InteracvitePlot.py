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
        color=df["Cluster"].astype(str),
        hover_data=["file_name"],
        category_orders={"Cluster": sorted(df["Cluster"].unique())},
        labels={"color": "Cluster"}
    )

    fig.update_traces(
        marker=dict(size=8),
        customdata=df["file_name"],  # Ensure file_name is passed to selections
        hovertemplate=None
    )

    # Enable selection mode in layout
    fig.update_layout(dragmode="select")

    app.layout = html.Div([
        dcc.Graph(id='scatter-plot', figure=fig, clear_on_unhover=True),

        dcc.Tooltip(id="graph-tooltip", direction="bottom"),

        dcc.Store(id="selected-points", data=[]),  # Store selected points

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
        direction = "bottom" if y > df["PCA2"].median() else "top"

        return True, bbox, children, direction

    @app.callback(
        Output("selected-points", "data"),
        Input("scatter-plot", "clickData"),
        State("selected-points", "data")
    )
    def update_selected_points(clickData, stored_points):
        print("Raw clickData:", clickData)  # Debugging output

        if clickData is None or "points" not in clickData:
            print("No point clicked. Returning current selection:", stored_points)
            return stored_points  # Don't reset on empty clickData

        # Extract file name from clicked point
        file_name = clickData["points"][0]["customdata"]

        # Ensure stored_points is a list
        if not isinstance(stored_points, list):
            stored_points = []

        # Update selection, keeping only last 2
        if file_name not in stored_points:
            stored_points.append(file_name)

        if len(stored_points) > 2:
            stored_points = stored_points[-2:]

        print("Updated selected points:", stored_points)
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

        img1_src = f"/assets/pngs/{selected_files[0]}.png" if len(selected_files) > 0 else ""
        img2_src = f"/assets/pngs/{selected_files[1]}.png" if len(selected_files) > 1 else ""

        img1_style = {"width": "100%", "display": "block"} if img1_src else {"display": "none"}
        img2_style = {"width": "100%", "display": "block"} if img2_src else {"display": "none"}

        return img1_src, img1_style, img2_src, img2_style

    return app


script_path = Path(__file__).resolve()
project_root = script_path.parents[0]

df = pd.read_csv(project_root / "pcaDF.csv")
app = create_dash_app(df, project_root)
app.run(debug=True)
