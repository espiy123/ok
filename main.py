import base64
import random
import dash
import networkx as nx
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
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

def aco(graph, iterations=100, n_ants=10, alpha=1, beta=5, evaporation=0.5, Q=100):
    nodes = list(graph)
    distances = {(i, j): euc(*graph[i], *graph[j]) for i in nodes for j in nodes if i != j}
    pheromones = {k: 1.0 for k in distances}
    best_path, best_length = None, float('inf')

    for _ in range(iterations):
        all_paths = []
        for _ in range(n_ants):
            path = [random.choice(nodes)]
            while len(path) < len(nodes):
                current = path[-1]
                unvisited = [n for n in nodes if n not in path]
                probabilities = [
                    (n, (pheromones[(current, n)] ** alpha) * (1 / distances[(current, n)] ** beta))
                    for n in unvisited
                ]
                total = sum(p for _, p in probabilities) or len(probabilities)
                r = random.uniform(0, total)
                acc = 0
                for node, prob in probabilities:
                    acc += prob
                    if acc >= r:
                        path.append(node)
                        break
            path.append(path[0])
            length = sum(distances[(path[i], path[i+1])] for i in range(len(path)-1))
            all_paths.append((path, length))
            if length < best_length:
                best_path, best_length = path, length

        for k in pheromones:
            pheromones[k] *= (1 - evaporation)
        for path, length in all_paths:
            for i in range(len(path) - 1):
                pheromones[(path[i], path[i + 1])] += Q / length
                pheromones[(path[i + 1], path[i])] += Q / length

    return best_path, best_length

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
    State('param-q', 'value')
)
def update_graph(_, graph, algorithm, iters, ants, alpha, beta, evap, q):
    if not graph:
        return go.Figure()

    path, length = greedy(graph) if algorithm == 'greedy' else aco(graph, iters, ants, alpha, beta, evap, q)

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
