import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.linear_node = nn.Linear(input_dim, output_dim)
        self.linear_edge = nn.Linear(input_dim, output_dim)
        self.attention = nn.Linear(output_dim, 1)

    def forward(self, node_features, edge_features, adjacency_matrix):
        # Aggregate node and edge features
        node_transformed = self.linear_node(node_features)
        edge_transformed = self.linear_edge(edge_features)
        aggregated = torch.tanh(node_transformed + edge_transformed)

        # Neighborhood weighted average
        attention_scores = self.attention(aggregated)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_aggregated = torch.sum(aggregated * attention_weights, dim=1)

        # Residual connection
        output = node_features + weighted_aggregated
        return output



class HighOrderPairwiseInteraction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HighOrderPairwiseInteraction, self).__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, ligand_features, receptor_features):
        # Generate pairwise interactions
        n_ligand = ligand_features.size(0)
        n_receptor = receptor_features.size(0)
        interactions = torch.zeros(n_ligand, n_receptor, self.linear.out_features)

        for i in range(n_ligand):
            for j in range(n_receptor):
                pair_features = torch.cat((ligand_features[:, i], receptor_features[:, j]), dim=0)
                interactions[i, j] = self.linear(pair_features)

        return interactions

class ProteinInterfacePrediction(nn.Module):
    def __init__(self, gnn_input_dim, gnn_output_dim, hopi_output_dim, cnn_channels):
        super(ProteinInterfacePrediction, self).__init__()
        self.gnn = GNNLayer(gnn_input_dim, gnn_output_dim)
        self.hopi = HighOrderPairwiseInteraction(gnn_output_dim, hopi_output_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(hopi_output_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, 1, kernel_size=1)
        )

    def forward(self, ligand_node_features, ligand_edge_features, ligand_adjacency, receptor_node_features, receptor_edge_features, receptor_adjacency, ligand_order, receptor_order):
        # GNN for ligand and receptor
        ligand_features = self.gnn(ligand_node_features, ligand_edge_features, ligand_adjacency)
        receptor_features = self.gnn(receptor_node_features, receptor_edge_features, receptor_adjacency)


        # High-order pairwise interactions
        interactions = self.hopi(ligand_features, receptor_features)

        # 2D CNN for dense prediction
        interactions = interactions.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
        predictions = self.cnn(interactions)
        predictions = predictions.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

        return predictions