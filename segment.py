import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models.segmentation as models
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from torch.utils.tensorboard import SummaryWriter

# Custom dataset for segmentation
class SegmentationDataset(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, transform=None, mask_transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        mask_path = os.path.join(self.mask_dir, self.data.iloc[idx, 1])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# Transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mask_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset
def load_segmentation_data():
    dataset = SegmentationDataset(
        csv_file='./images/metadata.csv',
        image_dir='./images/Image',
        mask_dir='./images/Mask',
        transform=image_transforms,
        mask_transform=mask_transforms
    )

    # Split dataset (70% train, 15% val, 15% test)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


# Load and prepare a segmentation model
def get_model(num_classes):
    model = models.fcn_resnet50(weights="DEFAULT")  # Fully Convolutional Network

    # Modify the final classifier layer for segmentation
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)  # Change output channels to match num_classes

    model = model.to(device)
    return model


# Dice loss (alternative to CrossEntropyLoss)
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

# Training function for segmentation
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float("inf")
    writer = SummaryWriter(log_dir='runs/segment_experiment')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (images, masks) in enumerate(train_loop):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)["out"]
            loss = criterion(outputs, masks.squeeze(1).long())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

            # Log training loss
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        iou_scores = []

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)["out"]
                loss = criterion(outputs, masks.squeeze(1).long())

                val_loss += loss.item()

                # Convert outputs to binary mask (thresholding at 0.5)
                #preds = torch.sigmoid(outputs) > 0.5
                preds = outputs.argmax(dim=1)  # Convert logits to class indices
                masks = masks > 0.5

                # Compute IoU (Intersection over Union)
                iou = jaccard_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')
                iou_scores.append(iou)

        avg_val_loss = val_loss / len(val_loader)
        avg_iou = sum(iou_scores) / len(iou_scores)


        # Log validation loss & IoU
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('IoU/val', avg_iou, epoch)

        if epoch % 5 == 0:
            # Add a batch dimension if it's not already there
            print(preds.size())
            print(masks.size())
            writer.add_images('Predictions', preds.unsqueeze(0), epoch)
            writer.add_images('Ground Truth', masks, epoch)



        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/segmentation_model.pth')
            print("Saved best model.")

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, test_loader = load_segmentation_data()

# Get model
num_classes = 2  # Change this based on your dataset (e.g., 1 for binary segmentation, >1 for multi-class)
model = get_model(num_classes)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
num_epochs = 30
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
