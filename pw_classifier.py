import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

import os
import gzip
import pickle
import urllib.request
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

# Predefined URLs for downloading the datasets
DATA_URLS = {
    "train": "https://zenodo.org/records/1127774/files/train.cpkl.gz?download=1",
    "test": "https://zenodo.org/records/1127774/files/test.cpkl.gz?download=1"
}

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
        split: "train" or "test"
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


# Create datasets
train_dataset = GraphDataset("train")
test_dataset = GraphDataset("test")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Example usage


# Define dataset class
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
        num_residues = hood_indices.size(0)
        adjacency = torch.zeros(num_residues, num_residues)
        for i in range(num_residues):
            for j in range(hood_indices.size(1)):
                neighbor_idx = hood_indices[i, j].item()
                adjacency[i, neighbor_idx] = 1
        return adjacency

# Define GNN layer
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

# Define high-order pairwise interaction layer
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
                pair_features = torch.cat((ligand_features[i], receptor_features[j]), dim=0)
                interactions[i, j] = self.linear(pair_features)

        return interactions

# Define protein interface prediction model
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

    def forward(self, ligand_node_features, ligand_edge_features, ligand_adjacency, receptor_node_features, receptor_edge_features, receptor_adjacency):
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

# Define training function
def train(model, dataloader, optimizer, criterion, device):
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

# Define evaluation function
def evaluate(model, dataloader, criterion, device):
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

# Main function
def main():
    # Example dataset
    data = [
        ["1A2B", "3C4D"],  # List of PDB codes
        [
            {
                "r_vertex": np.random.rand(100, 70),  # Receptor vertex features
                "l_vertex": np.random.rand(80, 70),   # Ligand vertex features
                "l_edge": np.random.rand(80, 20, 2),  # Ligand edge features
                "r_edge": np.random.rand(100, 20, 2), # Receptor edge features
                "l_hood_indices": np.random.randint(0, 80, (80, 20, 1)),  # Ligand neighborhood indices
                "r_hood_indices": np.random.randint(0, 100, (100, 20, 1)), # Receptor neighborhood indices
                "label": np.random.randint(-1, 2, (8000, 3))  # Labels
            },
            # More data...
        ]
    ]

    # Create dataset and dataloader
    dataset = ProteinInterfaceDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinInterfacePrediction(gnn_input_dim=70, gnn_output_dim=128, hopi_output_dim=64, cnn_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, dataloader, optimizer, criterion, device)
        val_loss = evaluate(model, dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()