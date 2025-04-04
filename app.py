# app.py

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from utils.data_preprocessing import load_and_preprocess_data
from utils.model_utils import train_svm_model, evaluate_model

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and preprocess the Titanic dataset
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('titanic.csv')

# App layout
app.layout = html.Div(children=[
    html.H1('Titanic Survival Prediction (SVM Dashboard)'),

    html.Div([
        html.Label('Kernel'),
        dcc.Dropdown(
            id='kernel',
            options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'RBF', 'value': 'rbf'},
                {'label': 'Poly', 'value': 'poly'},
                {'label': 'Sigmoid', 'value': 'sigmoid'}
            ],
            value='rbf'  # default value
        ),

        html.Label('C (Regularization parameter)'),
        dcc.Slider(
            id='C',
            min=0.1,
            max=10,
            step=0.1,
            value=1.0,
            marks={i: str(i) for i in range(1, 11)}
        ),

        html.Label('Gamma (Kernel coefficient)'),
        dcc.Slider(
            id='gamma',
            min=0.01,
            max=1,
            step=0.01,
            value=0.1,
            marks={0.01: '0.01', 0.5: '0.5', 1: '1'}
        )
    ], style={'width': '40%', 'display': 'inline-block', 'padding': '20px'}),

    html.Div(id='output-metrics', style={'margin-top': '20px'}),

    dcc.Graph(id='svm-graph')
])


# Callback to update graph and metrics based on user input
@app.callback(
    Output('svm-graph', 'figure'),
    Output('output-metrics', 'children'),
    Input('kernel', 'value'),
    Input('C', 'value'),
    Input('gamma', 'value')
)
def update_graph(kernel, C, gamma):
    # Train the model using selected parameters
    model = train_svm_model(X_train, y_train, kernel=kernel, C=C, gamma=gamma)

    # Evaluate model performance
    accuracy, cm = evaluate_model(model, X_test, y_test)

    # Prepare mesh grid for decision boundary visualization
    h = .02  # step size
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create plotly figure
    figure = go.Figure()

    # Add decision boundary
    figure.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        showscale=False,
        colorscale='RdBu',
        opacity=0.4
    ))

    # Add training points
    figure.add_trace(go.Scatter(
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

    figure.update_layout(
        title='SVM Decision Boundary',
        xaxis_title='Age (scaled)',
        yaxis_title='Fare (scaled)',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Prepare model metrics output
    metrics_text = [
        html.P(f'Accuracy: {accuracy:.2f}'),
        html.P(f'Confusion Matrix:'),
        html.Pre(f'{cm}')
    ]

    return figure, metrics_text


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
