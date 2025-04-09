# classification-demo/app.py

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from utils.data_preprocessing import load_and_preprocess_data
from utils.model_utils import train_svm_model, evaluate_model, compute_roc
import dash
import dash_bootstrap_components as dbc
app = dash.Dash(external_stylesheets=[dbc.themes.LUMEN])

# Define default parameters
DEFAULT_KERNEL = 'rbf'
DEFAULT_C = 1.0
DEFAULT_GAMMA = 0.1

# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('/Users/laurenlanda/PycharmProjects/TBoDLab5/data/titanic.csv')

# App layout
app.layout = html.Div(children=[

    # Title and subtitle
    html.H1('Titanic Survival Classification Dashboard'),
    html.H3('Class: CS-150 | Author: Lauren Landa'),

    html.Div([

        # Kernel selection
        html.Label('Kernel'),
        dcc.Dropdown(
            id='kernel',
            options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'RBF', 'value': 'rbf'},
                {'label': 'Poly', 'value': 'poly'},
                {'label': 'Sigmoid', 'value': 'sigmoid'}
            ],
            value=DEFAULT_KERNEL
        ),

        # Regularization parameter C
        html.Label('C (Regularization parameter)'),
        dcc.Slider(
            id='C',
            min=0.1,
            max=10,
            step=0.1,
            value=DEFAULT_C,
            marks={i: str(i) for i in range(1, 11)}
        ),

        # Gamma parameter
        html.Label('Gamma (Kernel coefficient)'),
        dcc.Slider(
            id='gamma',
            min=0.01,
            max=10,
            step=0.01,
            value=DEFAULT_GAMMA,
            marks={0.01: '0.01', 0.5: '0.5', 1: '1'}
        ),

        # Reset button
        html.Button('Reset Parameters', id='reset-button', n_clicks=0, style={'margin-top': '20px'})

    ], style={'width': '40%', 'display': 'inline-block', 'padding': '20px'}),

    # Metrics output
    html.Div(id='output-metrics', style={'margin-top': '20px'}),

    # Graphs: Decision boundary, ROC curve
    dcc.Graph(id='svm-graph'),
    dcc.Graph(id='roc-curve'),

    # Confusion matrix table
    html.H4('Confusion Matrix'),
    html.Div(id='confusion-matrix')

])

# Callback to reset parameters
@app.callback(
    Output('kernel', 'value'),
    Output('C', 'value'),
    Output('gamma', 'value'),
    Input('reset-button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_parameters(n_clicks):
    return DEFAULT_KERNEL, DEFAULT_C, DEFAULT_GAMMA

# Main callback to update graph and metrics
@app.callback(
    Output('svm-graph', 'figure'),
    Output('output-metrics', 'children'),
    Output('roc-curve', 'figure'),
    Output('confusion-matrix', 'children'),
    Input('kernel', 'value'),
    Input('C', 'value'),
    Input('gamma', 'value')
)
def update_outputs(kernel, C, gamma):
    # Train model with current parameters
    model = train_svm_model(X_train, y_train, kernel=kernel, C=C, gamma=gamma)

    # Evaluate model: accuracy, confusion matrix
    accuracy, cm, y_score, y_pred = evaluate_model(model, X_test, y_test)

    # Prepare decision boundary plot
    h = .02  # step size for mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    decision_boundary_fig = go.Figure()

    decision_boundary_fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        showscale=False,
        colorscale='RdBu',
        opacity=0.4
    ))

    decision_boundary_fig.add_trace(go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode='markers',
        marker=dict(
            color=y_train,
            colorscale='Viridis',
            line=dict(width=1),
            size=8
        ),
        name='Training Data'
    ))

    decision_boundary_fig.update_layout(
        title='Decision Boundary (Age vs. Fare)',
        xaxis_title='Age (scaled)',
        yaxis_title='Fare (scaled)',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Prepare ROC curve
    fpr, tpr, auc_score = compute_roc(y_test, y_score)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.2f})'
    ))
    roc_fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash'),
        showlegend=False
    ))
    roc_fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Prepare confusion matrix table
    cm_df = pd.DataFrame(cm, index=['Actual: False', 'Actual: True'], columns=['Predicted: False', 'Predicted: True'])
    confusion_matrix_table = html.Table([
        html.Thead(
            html.Tr([html.Th()] + [html.Th(col) for col in cm_df.columns])
        ),
        html.Tbody([
            html.Tr([html.Th(idx)] + [html.Td(cm_df.loc[idx, col]) for col in cm_df.columns])
            for idx in cm_df.index
        ])
    ], style={'width': '50%', 'margin': 'auto', 'border': '1px solid black'})

    # Prepare metrics text
    metrics_text = [
        html.P(f'Accuracy: {accuracy:.2f}')
    ]

    return decision_boundary_fig, metrics_text, roc_fig, confusion_matrix_table

# Run app
if __name__ == '__main__':
    app.run(debug=True)
