import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Initialize TensorBoard writer
log_dir = os.path.join('runs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
writer = SummaryWriter(log_dir)

# Define hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization for training
# Just normalization for validation and testing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Custom ImageFolder dataset to track filenames
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return sample, target, path

# Create dataset from image folders
def load_split_data():
    # Load all images with path tracking
    full_dataset = ImageFolderWithPaths(root='./images/')
    
    # Get class names
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    
    # Calculate splits (70% train, 15% val, 15% test)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Generate indices for the splits
    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Apply transformations using a custom collate function
    def train_collate_fn(batch):
        samples = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        paths = [item[2] for item in batch]
        
        # Apply transformations
        transformed_samples = [data_transforms['train'](sample) for sample in samples]
        
        return torch.stack(transformed_samples), torch.tensor(targets), paths
    
    def val_collate_fn(batch):
        samples = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        paths = [item[2] for item in batch]
        
        # Apply transformations
        transformed_samples = [data_transforms['val'](sample) for sample in samples]
        
        return torch.stack(transformed_samples), torch.tensor(targets), paths
    
    def test_collate_fn(batch):
        samples = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        paths = [item[2] for item in batch]
        
        # Apply transformations
        transformed_samples = [data_transforms['test'](sample) for sample in samples]
        
        return torch.stack(transformed_samples), torch.tensor(targets), paths
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=val_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=test_collate_fn)
    
    return train_loader, val_loader, test_loader, class_names

# Load and prepare the pre-trained ConvNext model
def get_model(num_classes):
    # Load pre-trained ConvNext model
    model = models.convnext_base(weights='IMAGENET1K_V1')
    
    # Freeze early layers
    for param in list(model.parameters())[:-30]:  # Adjust freezing as needed
        param.requires_grad = False
    
    # Fix the classifier structure to match ConvNext's expected dimensions
    # ConvNext expects a specific classifier structure
    model.classifier = nn.Sequential(
        # Global average pooling is already in the model's forward pass
        # Flatten from [B, C, 1, 1] to [B, C]
        nn.Flatten(1),
        # Then apply layer norm to the flattened tensor
        nn.LayerNorm(1024),
        nn.Dropout(0.2),
        nn.Linear(1024, num_classes)
    )
    
    model = model.to(device)
    return model

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, class_names):
    best_val_acc = 0.0
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels, _ in train_loop:  # Ignore paths during training
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            train_loop.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels, _ in val_loop:  # Ignore paths during validation
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_accs.append(val_acc)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_val_labels, all_val_preds)
        cm_figure = plot_confusion_matrix(cm, class_names)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        writer.add_figure('Confusion Matrix/validation', cm_figure, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Confusion Matrix:\n{cm}')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model.pth')
            print(f"Saved new best model with validation accuracy: {best_val_acc:.4f}")
    
    return train_accs, val_accs

# Evaluate the model on test set with filename logging
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for inputs, labels, paths in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
    
    test_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    cm_figure = plot_confusion_matrix(cm, class_names)
    
    # Log to TensorBoard
    writer.add_scalar('Accuracy/test', test_acc, 0)
    writer.add_figure('Confusion Matrix/test', cm_figure, 0)
    
    # Create a detailed error log with filenames
    results = []
    for true_label, pred_label, path in zip(all_labels, all_preds, all_paths):
        filename = os.path.basename(path)
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        is_correct = true_label == pred_label
        
        results.append({
            'filename': filename,
            'path': path,
            'true_class': true_class,
            'predicted_class': pred_class,
            'is_correct': is_correct
        })
    
    # Create a DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    
    # Save all results to CSV
    results_df.to_csv('test_results.csv', index=False)
    
    # Group images by confusion matrix categories
    cm_categories = {}
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            key = f"{true_class}_predicted_as_{pred_class}"
            files = results_df[(results_df['true_class'] == true_class) & 
                              (results_df['predicted_class'] == pred_class)]['filename'].tolist()
            cm_categories[key] = files
    
    # Save categorized results
    with open('confusion_matrix_files.txt', 'w') as f:
        for category, files in cm_categories.items():
            if files:  # Only write non-empty categories
                f.write(f"{category} ({len(files)} files):\n")
                for file in files:
                    f.write(f"  - {file}\n")
                f.write("\n")
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Detailed results saved to 'test_results.csv'")
    print(f"Confusion matrix categorized files saved to 'confusion_matrix_files.txt'")
    
    return test_acc

# Plot the training and validation accuracy curves
def plot_accuracy_curves(train_accs, val_accs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accs) + 1), train_accs, 'b-', label='Training Accuracy')
    plt.plot(range(1, len(val_accs) + 1), val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curves.png')
    plt.close()
    
    # Also add the figure to TensorBoard
    img = plt.imread('accuracy_curves.png')
    writer.add_image('Accuracy Curves', np.transpose(img, (2, 0, 1)), 0)

# Main execution
def main():
    print(f"Using device: {device}")
    
    # Load and split data
    train_loader, val_loader, test_loader, class_names = load_split_data()
    print(f"Dataset loaded with paths tracking")
    
    # Create model
    num_classes = len(class_names)
    model = get_model(num_classes)
    print(f"Model created: ConvNext Base with {num_classes} output classes")
    
    # Skip adding model graph to TensorBoard to avoid the error
    # Comment out or remove the following lines:
    # dummy_input = torch.randn(1, 3, 224, 224).to(device)
    # writer.add_graph(model, dummy_input)
    print("Skipping model graph visualization to avoid dimension mismatch error")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Train the model
    print("Starting training...")
    train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, class_names)
    
    # Plot accuracy curves
    plot_accuracy_curves(train_accs, val_accs)
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('model.pth'))
    test_acc = evaluate_model(model, test_loader, class_names)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': class_names,
        'test_accuracy': test_acc
    }, 'model.pth')
    
    print("Training completed. Model saved to 'model.pth'")
    writer.close()

if __name__ == "__main__":
    main()