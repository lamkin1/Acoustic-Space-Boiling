import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import os
from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# Helper for creating camera-rotation animation on a 3D Plotly scene
# -----------------------------------------------------------------------------
def add_camera_rotation(fig, axis='y', steps=120, radius=1.8, height=1.2):
    """
    Given a Plotly Figure `fig` with a 3D scene, add `steps` frames
    rotating the camera around the specified `axis` and wire up a Play button.
    """
    frames = []
    angles = np.linspace(0, 360, steps, endpoint=False)
    for i, angle in enumerate(angles):
        theta = np.radians(angle)
        if axis == 'y':
            eye = dict(x= radius * np.sin(theta),
                       y= radius * np.cos(theta),
                       z= height)
        elif axis == 'x':
            eye = dict(x= height,
                       y= radius * np.sin(theta),
                       z= radius * np.cos(theta))
        else:  # rotate around Z
            eye = dict(x= radius * np.cos(theta),
                       y= radius * np.sin(theta),
                       z= height)
        frames.append(dict(
            name=str(i),
            layout=dict(scene_camera=dict(eye=eye))
        ))

    # Set initial zoom (camera) to be closer for a "zoomed-in" effect
    fig.update_layout(scene_camera=dict(eye=dict(x=radius, y=0, z=height)))

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
                pad=dict(t=60, r=10),
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


def create_dash_app(results_dict, directory):
    app = dash.Dash(__name__)

    # Load human labels
    labels_df = pd.read_csv(directory / "labels.csv")
    labels_df["file_name"] = labels_df["file_name"].str.strip().apply(lambda s: "MATLAB " + s)

    label_mapping = {
        1: "Single Rhythmic",
        2: "Double Rhythmic",
        3: "Random",
        4: "Rhythmic with Climax",
        5: "Noise",
        6: "1 Rhythmic with Random",
        7: "Triple Rhythmic",
        8: "Transition"
    }
    labels_df["label_desc"] = labels_df["label"].map(label_mapping)

    # Dropdown configs
    pca_count_options = [{'label': str(n), 'value': n} for n in range(2, 7)]
    clusters_options = [{'label': str(n), 'value': n} for n in range(2, 8)]
    mode_options = [
        {'label': 'Clustering Results', 'value': 'clustering'},
        {'label': 'Human Labels', 'value': 'labels'}
    ]

    default_pca = 2
    default_clusters = 2
    axis_options = [{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, default_pca+1)]

    # Default 2D scatter
    dff0 = results_dict[default_clusters]
    default_fig = px.scatter(
        dff0,
        x="PCA1",
        y="PCA2",
        color=dff0[f'Cluster_{default_pca}_PCs'].astype(str),
        hover_data=["file_name"],
        category_orders={f'Cluster_{default_pca}_PCs': sorted(dff0[f'Cluster_{default_pca}_PCs'].unique())},
        labels={"color": "Cluster"}
    )
    default_fig.update_traces(marker=dict(size=8), customdata=dff0["file_name"], hovertemplate=None)
    default_fig.update_layout(dragmode="select")

    # Layout
    app.layout = html.Div([
        html.Div([
            html.Div([html.Label("Visualization Mode:"), dcc.Dropdown(id="viz-mode", options=mode_options, value="clustering")], style={"width":"30%","padding":"10px"}),
            html.Div([html.Label("Number of PCA Components:"), dcc.Dropdown(id="pca-count", options=pca_count_options, value=default_pca)], style={"width":"30%","padding":"10px"}),
            html.Div([html.Label("Number of Clusters:"), dcc.Dropdown(id="clusters-count", options=clusters_options, value=default_clusters)], style={"width":"30%","padding":"10px"})
        ], style={"display":"flex","justify-content":"space-around"}),

        html.Div([
            html.Div([html.Label("X-Axis:"), dcc.Dropdown(id="xaxis", options=axis_options, value="PCA1")], style={"width":"30%","padding":"10px"}),
            html.Div([html.Label("Y-Axis:"), dcc.Dropdown(id="yaxis", options=axis_options, value="PCA2")], style={"width":"30%","padding":"10px"}),
            html.Div([html.Label("Z-Axis:"), dcc.Dropdown(id="zaxis", options=[], value="PCA3")], id="zaxis-div", style={"display":"none","width":"30%","padding":"10px"})
        ], style={"display":"flex","justify-content":"space-around"}),

        dcc.Store(id="current-yaxis", data="PCA2"),
        dcc.Graph(id='scatter-plot', figure=default_fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip", direction="bottom"),
        dcc.Store(id="selected-points", data=[]),

        html.Div(id="comparison-section", children=[
            html.H3("Comparison View", style={"text-align":"center","margin-top":"20px"}),
            html.Div([
                html.Div([html.Img(id="img1", style={"width":"100%","display":"none"})], style={"width":"40%","padding":"10px","text-align":"center"}),
                html.Div([html.Img(id="img2", style={"width":"100%","display":"none"})], style={"width":"40%","padding":"10px","text-align":"center"})
            ], style={"display":"flex","justify-content":"center","align-items":"center"})
        ])
    ])

    # Callbacks
    @app.callback(
        Output("xaxis", "options"),
        Output("yaxis", "options"),
        Output("zaxis", "options"),
        Input("pca-count", "value")
    )
    def update_axis_options(pca_count):
        opts = [{'label': f'PCA{i}', 'value': f'PCA{i}'} for i in range(1, pca_count+1)]
        return opts, opts, (opts if pca_count>2 else [])

    @app.callback(
        Output("zaxis-div", "style"),
        Input("pca-count", "value")
    )
    def toggle_zaxis(pca_count):
        return {"display":"inline-block","width":"30%","padding":"10px"} if pca_count>2 else {"display":"none"}

    @app.callback(
        Output("scatter-plot", "figure"),
        Output("current-yaxis", "data"),
        Input("xaxis","value"),
        Input("yaxis","value"),
        Input("zaxis","value"),
        Input("pca-count","value"),
        Input("clusters-count","value"),
        Input("viz-mode","value")
    )
    def update_scatter(xaxis, yaxis, zaxis, pca_count, clusters_count, viz_mode):
        dff = results_dict[clusters_count]
        if viz_mode == 'labels':
            dff = dff.merge(labels_df, on="file_name", how="left")
            color_col = 'label_desc'; color_label='Regime'
        else:
            color_col = f'Cluster_{pca_count}_PCs'; color_label='Cluster'

        if pca_count > 2:
            fig = px.scatter_3d(
                dff, x=xaxis, y=yaxis, z=zaxis,
                color=dff[color_col].astype(str),
                hover_data=["file_name"],
                category_orders={color_col: sorted(dff[color_col].astype(str).unique())},
                labels={"color": color_label}
            )
            # inject autorotate
            fig = add_camera_rotation(fig, axis='y', steps=120, radius=1.8, height=0.5)
        else:
            fig = px.scatter(
                dff, x=xaxis, y=yaxis,
                color=dff[color_col].astype(str),
                hover_data=["file_name"],
                category_orders={color_col: sorted(dff[color_col].astype(str).unique())},
                labels={"color": color_label}
            )

        fig.update_traces(marker=dict(size=8), customdata=dff["file_name"], hovertemplate=None)
        fig.update_layout(dragmode="select")
        return fig, yaxis

    @app.callback(
        Output("graph-tooltip","show"),
        Output("graph-tooltip","bbox"),
        Output("graph-tooltip","children"),
        Output("graph-tooltip","direction"),
        Input("scatter-plot","hoverData"),
        Input("current-yaxis","data")
    )
    def display_hover(hoverData, current_yaxis):
        if not hoverData: return False, dash.no_update, dash.no_update, dash.no_update
        point = hoverData["points"][0]
        basename = point["customdata"] + ".png"
        imgp = directory / "assets" / basename
        if not imgp.exists(): return False, dash.no_update, dash.no_update, dash.no_update
        bbox = point["bbox"]
        children = html.Div([html.Img(src=f"/assets/{basename}", style={"width":"200px"})])
        direction = "bottom" if point["y"] > 0 else "top"
        return True, bbox, children, direction

    @app.callback(
        Output("selected-points","data"),
        Input("scatter-plot","clickData"),
        State("selected-points","data")
    )
    def update_selected(clickData, stored):
        if not clickData or 'points' not in clickData: return stored
        fn = clickData['points'][0]['customdata']
        stored = stored or []
        if fn not in stored: stored.append(fn)
        return stored[-2:]

    @app.callback(
        Output("img1","src"),Output("img1","style"),
        Output("img2","src"),Output("img2","style"),
        Input("selected-points","data")
    )
    def show_images(sel):
        if not sel: return "",{"display":"none"},"",{"display":"none"}
        s1 = f"/assets/{sel[0]}.png" if len(sel)>0 else ""
        s2 = f"/assets/{sel[1]}.png" if len(sel)>1 else ""
        return s1,{"width":"100%","display":"block"} if s1 else {"display":"none"}, s2,{"width":"100%","display":"block"} if s2 else {"display":"none"}

    return app

# -----------------------------------------------------------------------------
# App bootstrap
# -----------------------------------------------------------------------------
script_path = Path(__file__).resolve()
project_root = script_path.parents[0]

# Read CSVs
results_dict = {}
for k in range(2,8):
    df = pd.read_csv(project_root / f"assets/results_{k}_clusters.csv")
    df["file_name"] = df["file_name"].str[15:]
    results_dict[k] = df

app = create_dash_app(results_dict, project_root)

# Simple password auth
main_layout = app.layout
auth_layout = html.Div([
    html.H2("Please Log In"),
    dcc.Input(id='password-input', type='password', placeholder='Enter Password'),
    html.Button("Submit", id='login-button'),
    html.Div(id='login-output', style={'color':'red'})
], style={'text-align':'center','margin-top':'100px'})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='logged-in', storage_type='session'),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content','children'),
    Input('logged-in','data')
)
def render(logged):
    return main_layout

@app.callback(
    Output('logged-in','data'),
    Output('login-output','children'),
    Input('login-button','n_clicks'),
    State('password-input','value'),
    prevent_initial_call=True
)
def verify(n, pw):
    if pw == "NASA_Boiling": return True, ""
    return dash.no_update, "Incorrect password. Please try again."

server = app.server

if __name__ == '__main__':
    app.run(debug=False)
