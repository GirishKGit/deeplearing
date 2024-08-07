import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import plotly.graph_objects as go
import altair as alt
import pandas as pd
from bokeh.layouts import column
from bokeh.io import output_file, show
from bokeh.plotting import figure

class Network(object):

    def __init__(self, sizes):
        """Initialize the neural network."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.writer = SummaryWriter()

    def sigmoid(self, x):
        """Compute the sigmoid of x"""
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Compute the derivative of the sigmoid of x"""
        return x * (1.0 - x)

    def feedforward(self, x):
        """Compute the output of the network for a single input."""
        activations = []
        weighted_sum = [np.dot(self.weights[i], x) + self.biases[i] for i in range(self.num_layers - 1)]
        activations = [self.sigmoid(sum) for sum in weighted_sum]
        return activations

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        """Train the neural network using mini-batch stochastic gradient descent."""
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                inputs = [item[0] for item in mini_batch]
                labels = [item[1] for item in mini_batch]
                self.weights[0] += eta * np.dot((self.feedforward(inputs[0])[1:] - labels[0]), np.transpose(inputs[0]))
                for i in range(1, self.num_layers - 1):
                    self.weights[i] += eta * np.dot((self.feedforward(inputs[0])[i+1:] - labels[0]), np.transpose(self.feedforward(inputs[0])[i]))
                    self.biases[i] += eta * (self.feedforward(inputs[0])[i+1:] - labels[0])
            if epoch % 10 == 0:
                losses = [sum((self.feedforward(x))[-1] - y) ** 2 for x, y in training_data]
                self.writer.add_scalar('Loss', np.mean(losses), epoch)

    def visualize_activations(self, activations, title="Layer Activations"):
        """Visualize activations using Plotly."""
        fig = go.Figure()
        for i, activation in enumerate(activations):
            fig.add_trace(go.Scatter(
                x=list(range(len(activation))),
                y=activation.flatten(),
                mode='markers',
                name=f'Layer {i+1}'
            ))
        fig.update_layout(title=title, xaxis_title='Neuron Index', yaxis_title='Activation')
        fig.show()

    def visualize_loss(self, losses, title="Training Loss"):
        """Visualize training loss using Altair."""
        df = pd.DataFrame({'Epoch': range(len(losses)), 'Loss': losses})
        chart = alt.Chart(df).mark_line().encode(
            x='Epoch',
            y='Loss'
        ).properties(title=title)
        chart.save("loss_chart.html")

    def visualize_weights(self, weights, title="Weight Histograms"):
        """Visualize weights using Bokeh."""
        plots = []
        for i, weight in enumerate(weights):
            hist, edges = np.histogram(weight.flatten(), bins=30)
            p = figure(title=f"{title} - Layer {i}", tools="save", 
                       x_axis_label='Weight Value', y_axis_label='Frequency')
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
            plots.append(p)
        return column(*plots)

    def visualize_biases(self, biases, title="Bias Histograms"):
        """Visualize biases using Bokeh."""
        plots = []
        for i, bias in enumerate(biases):
            hist, edges = np.histogram(bias.flatten(), bins=30)
            p = figure(title=f"{title} - Layer {i}", tools="save", 
                       x_axis_label='Bias Value', y_axis_label='Frequency')
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
            plots.append(p)
        return column(*plots)

# Example usage
if __name__ == "__main__":
    sizes = [784, 30, 10]
    net = Network(sizes)