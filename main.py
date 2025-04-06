import base64
import random
import dash
import networkx as nx
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import numpy as np
from numba import njit
from dash import html, dcc, Output, Input, State, callback_context

def euc(xa, ya, xb, yb):
    return ((xb - xa)**2 + (yb - ya)**2)**0.5

def greedy(graph):
    current = next(iter(graph))
    visited = [current]
    total_distance = 0.0
    unvisited = dict(graph)
    unvisited.pop(current)

    while unvisited:
        nearest, min_dist = min(
            ((node, euc(*graph[current], *coords)) for node, coords in unvisited.items()),
            key=lambda x: x[1]
        )
        visited.append(nearest)
        total_distance += min_dist
        current = nearest
        unvisited.pop(current)

    total_distance += euc(*graph[visited[-1]], *graph[visited[0]])
    visited.append(visited[0])
    return visited, total_distance

@njit
def euclidean_distance(xa, ya, xb, yb):
    return ((xb - xa)**2 + (yb - ya)**2)**0.5

@njit
def compute_aco_jit_elite(coords, iterations, n_ants, alpha, beta, evaporation, Q, elite_weight):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    pheromones = np.ones((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = euclidean_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])

    best_length = 1e10
    best_path = np.arange(n + 1)

    for _ in range(iterations):
        for _ in range(n_ants):
            path = np.full(n + 1, -1)
            visited = np.zeros(n, dtype=np.bool_)
            start = np.random.randint(0, n)
            path[0] = start
            visited[start] = True

            for step in range(1, n):
                current = path[step - 1]
                probs = np.zeros(n)
                denom = 0.0
                for j in range(n):
                    if not visited[j]:
                        probs[j] = (pheromones[current][j] ** alpha) * (1.0 / dist_matrix[current][j] ** beta)
                        denom += probs[j]

                if denom == 0:
                    continue

                r = np.random.random() * denom
                acc = 0.0
                for j in range(n):
                    if not visited[j]:
                        acc += probs[j]
                        if acc >= r:
                            path[step] = j
                            visited[j] = True
                            break

            path[-1] = path[0]
            length = 0.0
            for i in range(n):
                length += dist_matrix[path[i]][path[i+1]]

            if length < best_length:
                best_length = length
                best_path[:] = path

            for i in range(n):
                a = path[i]
                b = path[i+1]
                pheromones[a][b] += Q / length
                pheromones[b][a] += Q / length

        # ELITIST ANT feromone update
        for i in range(n):
            a = best_path[i]
            b = best_path[i+1]
            pheromones[a][b] += elite_weight * (Q / best_length)
            pheromones[b][a] += elite_weight * (Q / best_length)

        pheromones *= (1.0 - evaporation)

    return best_path, best_length

def aco(graph, iterations=100, n_ants=10, alpha=1, beta=5, evaporation=0.5, Q=100, elite_weight=1):
    node_list = list(graph)
    index_map = {node: idx for idx, node in enumerate(node_list)}
    coords = np.array([graph[node] for node in node_list])

    path_indices, best_length = compute_aco_jit_elite(coords, iterations, n_ants, alpha, beta, evaporation, Q, elite_weight)
    best_path = [node_list[i] for i in path_indices]

    return best_path, best_length

# --- DASH APP ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(fluid=True, style={'height': '100vh'}, children=[
    dbc.Row([
        dbc.Col(width=3, style={
            'maxHeight': '100vh',
            'overflowY': 'auto',
            'padding': '20px',
            'borderRight': '1px solid #ddd'
        }, children=[
            html.H4("Panel sterujący"),

            dcc.Upload(
                id='upload-data',
                children=html.Div(['Przeciągnij plik lub ', html.A('wybierz z dysku')]),
                style={
                    'height': '60px',
                    'lineHeight': '60px',
                    'border': '1px dashed #ccc',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-bottom': '20px'
                },
                multiple=False
            ),

            dcc.Dropdown(
                id='algorithm-choice',
                options=[
                    {'label': 'Greedy (najbliższy sąsiad)', 'value': 'greedy'},
                    {'label': 'Mrówkowy (ACO)', 'value': 'aco'}
                ],
                value='greedy',
                clearable=False,
                style={'margin-bottom': '20px'}
            ),

            html.Div(id='aco-params-container', children=[
                html.Label("Parametry ACO:"),

                *[
                    dbc.InputGroup([
                        dbc.InputGroupText(label),
                        dbc.Input(id=id_, type='number', value=val, step=step)
                    ], className="mb-2")
                    for label, id_, val, step in [
                        ("Liczba iteracji", "param-iterations", 100, 1),
                        ("Liczba mrówek", "param-ants", 10, 1),
                        ("Alpha", "param-alpha", 1.0, 0.1),
                        ("Beta", "param-beta", 5.0, 0.1),
                        ("Parowanie", "param-evaporation", 0.5, 0.01),
                        ("Q", "param-q", 100.0, 0.01),
                        ("Waga elitarnej mrówki", "param-elite", 1.0, 0.1)
                    ]
                ]
            ]),

            html.Label("Liczba punktów:"),
            dbc.Input(id='liczba-punktow', type='number', value=20, min=2, max=200, step=1, className='mb-3'),

            html.Div(id='file-info'),
            dbc.Button("Przelicz trasę", id='recalculate-btn', color='primary', className='mt-1 w-100'),
            dbc.Button("Generuj losowe punkty", id='generate-random-btn', color='secondary', className='mt-2 w-100'),
            html.Pre(id='coords-display', style={'whiteSpace': 'pre-wrap', 'marginTop': '10px'}),
            dcc.Store(id='graph-data')
        ]),

        dbc.Col(width=9, children=[
            dcc.Graph(id='graph-output', style={'height': '100vh'})
        ])
    ])
])

@app.callback(
    Output('aco-params-container', 'style'),
    Input('algorithm-choice', 'value')
)
def toggle_aco_params(algorithm):
    return {'display': 'block'} if algorithm == 'aco' else {'display': 'none'}

@app.callback(
    Output('graph-data', 'data'),
    Output('file-info', 'children'),
    Output('coords-display', 'children'),
    Input('generate-random-btn', 'n_clicks'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('liczba-punktow', 'value'),
    prevent_initial_call=True
)
def handle_input(n_clicks_generate, contents, filename, num_points):
    trigger = callback_context.triggered_id

    if contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string).decode('utf-8').splitlines()
            graph = {
                parts[0]: (float(parts[1]), float(parts[2]))
                for parts in (line.split() for line in decoded[1:int(decoded[0]) + 1])
            }
            info = f"Wczytano plik: {filename}"
        except Exception as e:
            return dash.no_update, f"Błąd przy wczytywaniu: {e}", dash.no_update

    elif trigger == 'generate-random-btn':
        graph = {
            str(i + 1): (random.randint(1, 10000), random.randint(1, 10000))
            for i in range(num_points)
        }
        info = f"Wygenerowano {num_points} losowych punktów"
    else:
        return dash.no_update, dash.no_update, dash.no_update

    coord_text = "\n".join([str(len(graph))] + [f"{k} {int(x)} {int(y)}" for k, (x, y) in graph.items()])
    return graph, info, coord_text

@app.callback(
    Output('graph-output', 'figure'),
    Input('recalculate-btn', 'n_clicks'),
    State('graph-data', 'data'),
    State('algorithm-choice', 'value'),
    State('param-iterations', 'value'),
    State('param-ants', 'value'),
    State('param-alpha', 'value'),
    State('param-beta', 'value'),
    State('param-evaporation', 'value'),
    State('param-q', 'value'),
    State('param-elite', 'value')
)
def update_graph(_, graph, algorithm, iters, ants, alpha, beta, evap, q, elite):
    if not graph:
        return go.Figure()

    path, length = greedy(graph) if algorithm == 'greedy' else aco(graph, iters, ants, alpha, beta, evap, q, elite)

    G = nx.Graph()
    G.add_nodes_from([(n, {'pos': pos}) for n, pos in graph.items()])
    G.add_edges_from((path[i], path[i + 1]) for i in range(len(path) - 1))

    pos = nx.get_node_attributes(G, 'pos')
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=2))
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(size=10)
    )

    return go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=f'Trasa - Długość: {round(length, 2)}',
        title_x=0.5,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    ))

if __name__ == '__main__':
    app.run(debug=True)
