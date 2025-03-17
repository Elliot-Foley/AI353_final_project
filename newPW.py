import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import gzip
import pickle
import urllib.request
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

# URLs for downloading the dataset
DATA_URLS = {
    "train": "https://zenodo.org/records/1127774/files/train.cpkl.gz?download=1",
    "test": "https://zenodo.org/records/1127774/files/test.cpkl.gz?download=1"
}

# Directory to store the downloaded data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the data directory exists


def download_file(url, dest_path):
    """Download a file from a URL with a progress bar."""
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists, skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")

    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as progress_bar:
            for data in iter(lambda: response.read(block_size), b""):
                out_file.write(data)
                progress_bar.update(len(data))
    print(f"Download complete: {dest_path}")


class GraphDataset(Dataset):
    def __init__(self, split):
        """
        Initialize the dataset.
        :param split: "train" or "test"
        """
        assert split in ["train", "test"], "Invalid dataset split. Choose 'train' or 'test'."
        self.file_path = os.path.join(DATA_DIR, f"{split}.cpkl.gz")

        # Download if file does not exist
        if not os.path.exists(self.file_path):
            download_file(DATA_URLS[split], self.file_path)

        self.data_list = self.load_cpkl()

    def load_cpkl(self):
        """Load compressed pickle file and return the list of graphs."""
        with gzip.open(self.file_path, 'rb') as f:
            return pickle.load(f, encoding='latin1')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """Return a PyTorch Geometric Data object for the given index."""
        return self.data_list[idx]


def collate_fn(batch):
    """Collate function for DataLoader to batch graph data."""
    return batch  # PyTorch Geometric's DataLoader will handle batching


# Define dataset class for protein interface prediction
class ProteinInterfaceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        # Extract features
        r_vertex = torch.tensor(self.data[1][idx]['r_vertex'], dtype=torch.float32)
        l_vertex = torch.tensor(self.data[1][idx]['l_vertex'], dtype=torch.float32)
        l_edge = torch.tensor(self.data[1][idx]['l_edge'], dtype=torch.float32)
        r_edge = torch.tensor(self.data[1][idx]['r_edge'], dtype=torch.float32)
        l_hood_indices = torch.tensor(self.data[1][idx]['l_hood_indices'], dtype=torch.long)
        r_hood_indices = torch.tensor(self.data[1][idx]['r_hood_indices'], dtype=torch.long)
        labels = torch.tensor(self.data[1][idx]['label'], dtype=torch.float32)

        # Create adjacency matrices
        l_adjacency = self.create_adjacency_matrix(l_hood_indices)
        r_adjacency = self.create_adjacency_matrix(r_hood_indices)

        return l_vertex, l_edge, l_adjacency, r_vertex, r_edge, r_adjacency, labels

    def create_adjacency_matrix(self, hood_indices):
        """Create an adjacency matrix from neighborhood indices."""
        num_residues = hood_indices.size(0)
        adjacency = torch.zeros(num_residues, num_residues)
        for i in range(num_residues):
            for j in range(hood_indices.size(1)):
                neighbor_idx = hood_indices[i, j].item()
                adjacency[i, neighbor_idx] = 1
        return adjacency


class GNNLayer(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, use_weighted_avg=False):
        """
        Initialize the GNN layer.
        :param node_feat_dim (int): Dimension of node features (d_N)
        :param edge_feat_dim (int): Dimension of edge features (d_E)
        :param use_weighted_avg (bool): Whether to use weighted average (NeiWA)
        """
        super(GNNLayer, self).__init__()
        self.use_weighted_avg = use_weighted_avg

        # Linear transformation matrices W_N and W_E
        self.W_N = nn.Linear(node_feat_dim, node_feat_dim)  # W_N: (d_N, d_N)
        self.W_E = nn.Linear(edge_feat_dim, node_feat_dim)  # W_E: (d_E, d_N)

        # If using weighted average, define a trainable vector q
        if use_weighted_avg:
            self.q = nn.Parameter(torch.randn(node_feat_dim))  # q: (d_N)

    def forward(self, node_features, edge_features):
        """
        Forward pass for the GNN layer.
        :param node_features (torch.Tensor): Neighbor node feature matrix Hi, shape (d_N, n)
        :param edge_features (torch.Tensor): Neighbor edge feature matrix Ei, shape (d_E, n)
        :return: Updated central node feature, shape (d_N)
        """
        # Formula (2): Zi = tanh(W_N * Hi + W_E * Ei)
        Zi = torch.tanh(self.W_N(node_features) + self.W_E(edge_features))  # Zi: (d_N, n)

        if self.use_weighted_avg:
            # Formula (4): a = softmax(Zi^T * q)
            a = F.softmax(torch.matmul(Zi.T, self.q), dim=0)  # a: (n)
            # Formula (5): h_i = h_i + (1/n) * Zi * a
            aggregated = torch.matmul(Zi, a) / Zi.size(1)  # aggregated: (d_N)
        else:
            # Formula (3): h_i = h_i + (1/n) * Zi * 1_n
            aggregated = torch.mean(Zi, dim=1)  # aggregated: (d_N)

        # Residual connection
        output = node_features[:, 0] + aggregated  # Central node feature + aggregated feature
        return output


class HighOrderPairwiseInteraction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HighOrderPairwiseInteraction, self).__init__()
        # Linear transformation for combining two node features
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, ligand_features, receptor_features):
        """
        Forward pass for high-order pairwise interaction.
        :param ligand_features: Ligand protein node features (batch_size, num_ligand_nodes, input_dim)
        :param receptor_features: Receptor protein node features (batch_size, num_receptor_nodes, input_dim)
        :return: High-order pairwise interaction 3D tensor (batch_size, num_ligand_nodes, num_receptor_nodes, output_dim)
        """
        batch_size, num_ligand_nodes, _ = ligand_features.size()
        _, num_receptor_nodes, _ = receptor_features.size()

        # Initialize 3D tensor
        pairwise_interactions = torch.zeros(batch_size, num_ligand_nodes, num_receptor_nodes, self.linear.out_features)

        # Iterate over all ligand and receptor node pairs
        for i in range(num_ligand_nodes):
            for j in range(num_receptor_nodes):
                # Combine features of two nodes
                combined = torch.cat((ligand_features[:, i, :], receptor_features[:, j, :]), dim=1)
                # Generate pairwise interaction features through linear transformation
                pairwise_interactions[:, i, j] = self.linear(combined)

        return pairwise_interactions


class CNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN, self).__init__()
        # 2D convolutional layers
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(output_channels, 1, kernel_size=1)  # Output single channel
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for CNN.
        :param x: Input tensor (batch_size, input_channels, height, width)
        :return: Output tensor (batch_size, 1, height, width)
        """
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x


class ProteinInterfacePrediction(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, gnn_output_dim, hopi_output_dim, cnn_output_dim, use_weighted_avg=False):
        super(ProteinInterfacePrediction, self).__init__()
        # Graph neural network layer
        self.gnn = GNNLayer(node_feat_dim, edge_feat_dim, use_weighted_avg)
        # High-order pairwise interaction layer
        self.hopi = HighOrderPairwiseInteraction(gnn_output_dim, hopi_output_dim)
        # 2D convolutional neural network
        self.cnn = CNN(hopi_output_dim, cnn_output_dim)

    def forward(self, ligand_node_features, ligand_edge_features, receptor_node_features, receptor_edge_features):
        """
        Forward pass for protein interface prediction.
        :param ligand_node_features: Ligand protein node features (batch_size, num_ligand_nodes, node_feat_dim)
        :param ligand_edge_features: Ligand protein edge features (batch_size, num_ligand_nodes, num_neighbors, edge_feat_dim)
        :param receptor_node_features: Receptor protein node features (batch_size, num_receptor_nodes, node_feat_dim)
        :param receptor_edge_features: Receptor protein edge features (batch_size, num_receptor_nodes, num_neighbors, edge_feat_dim)
        :return: Predicted interface matrix (batch_size, num_ligand_nodes, num_receptor_nodes)
        """
        # 1. Extract node features for ligand and receptor using GNN
        ligand_features = self.gnn(ligand_node_features, ligand_edge_features)
        receptor_features = self.gnn(receptor_node_features, receptor_edge_features)

        # 2. Generate high-order pairwise interaction 3D tensor
        pairwise_interactions = self.hopi(ligand_features, receptor_features)

        # 3. Reshape tensor for CNN input (batch_size, channels, height, width)
        pairwise_interactions = pairwise_interactions.permute(0, 3, 1, 2)

        # 4. Use 2D CNN for dense prediction
        output = self.cnn(pairwise_interactions)

        # 5. Convert output to probabilities using Sigmoid
        output = torch.sigmoid(output)

        return output.squeeze(1)  # Remove channel dimension


def train(model, dataloader, optimizer, criterion, device):
    """Training function."""
    model.train()
    total_loss = 0
    for batch in dataloader:
        l_vertex, l_edge, l_adjacency, r_vertex, r_edge, r_adjacency, labels = batch
        l_vertex, l_edge, l_adjacency = l_vertex.to(device), l_edge.to(device), l_adjacency.to(device)
        r_vertex, r_edge, r_adjacency = r_vertex.to(device), r_edge.to(device), r_adjacency.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(l_vertex, l_edge, l_adjacency, r_vertex, r_edge, r_adjacency)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluation function."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            l_vertex, l_edge, l_adjacency, r_vertex, r_edge, r_adjacency, labels = batch
            l_vertex, l_edge, l_adjacency = l_vertex.to(device), l_edge.to(device), l_adjacency.to(device)
            r_vertex, r_edge, r_adjacency = r_vertex.to(device), r_edge.to(device), r_adjacency.to(device)
            labels = labels.to(device)

            predictions = model(l_vertex, l_edge, l_adjacency, r_vertex, r_edge, r_adjacency)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    # Example dataset
    train_dataset = GraphDataset("train")
    test_dataset = GraphDataset("test")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Example usage
    for batch in train_loader:
        print(len(batch[1]))
        break
    for batch in train_loader:
        data = batch
        # Create dataset and dataloader
        dataset = ProteinInterfaceDataset(data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize model, optimizer, and loss function
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ProteinInterfacePrediction(node_feat_dim=70, edge_feat_dim=70, gnn_output_dim=70, hopi_output_dim=64,
                                           cnn_output_dim=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            train_loss = train(model, dataloader, optimizer, criterion, device)
            val_loss = evaluate(model, dataloader, criterion, device)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()