# app.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import os
from pathlib import Path

def create_dash_app(df, directory):
    """
    Parameters:
      df: DataFrame containing 'PCA1' ... 'PCA6', 'Cluster', and 'file_name'
      directory: Base directory (used to locate our CSV and image file)
    """
    # (No longer copying multiple PNG files since we'll use one fixed image.)

    # Create the Dash app
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

    # Define the layout of the app
    app.layout = html.Div([
        html.Div([
            html.Label("X-Axis:"),
            dcc.Dropdown(id="xaxis", options=pca_options, value="PCA1")
        ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),
        html.Div([
            html.Label("Y-Axis:"),
            dcc.Dropdown(id="yaxis", options=pca_options, value="PCA2")
        ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),
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
        # Instead of using file_name to build an image path,
        # we always use ex_image.png from assets/.
        image_path = "assets/ex_image.png"
        if not os.path.exists(image_path):
            return False, dash.no_update, dash.no_update, dash.no_update

        children = html.Div([
            html.Img(src=f"/{image_path}", style={"width": "200px"}),
        ])
        y = hover_data["y"]
        median_val = df[current_yaxis].median() if current_yaxis in df.columns else 0
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
        # Always use the same image for both comparison views.
        img_src = "/assets/ex_image.png"
        return img_src, {"width": "100%", "display": "block"}, img_src, {"width": "100%", "display": "block"}

    return app

# --- Initialize the app ---

from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parents[0]

# Load fake_data.csv from assets/ (adjust the path as needed)
df = pd.read_csv(project_root / "assets/fake_data.csv")

# (Optional: if your fake_data.csv's file_name column still has a prefix you wish to remove)
df["file_name"] = df["file_name"].str[64:]

app = create_dash_app(df, project_root)

# Export the underlying Flask server for production
server = app.server

if __name__ == '__main__':
    app.run(debug=True)
