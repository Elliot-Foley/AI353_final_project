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
for batch in train_loader:
   print(batch)  # This should print a batch of PyG Data objects
   break
