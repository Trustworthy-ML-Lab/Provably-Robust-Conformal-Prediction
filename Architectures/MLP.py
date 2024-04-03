import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_layers, in_features, out_features):
        super().__init__()
        features_now = in_features
        self.layers = nn.ModuleList()
        for n_neurons in hidden_layers:
            self.layers.append(nn.Sequential(nn.Linear(features_now, n_neurons), nn.BatchNorm1d(n_neurons)))
            features_now = n_neurons
        self.layers.append(nn.Linear(features_now, out_features))
    
    def forward(self, x):
        # Reshape x into a vector
        if len(x.shape) != 2:
            x = x.view((x.shape[0], -1))
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.functional.relu(x)
        return self.layers[-1](x)

