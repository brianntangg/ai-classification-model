"""
train.py — People vs Food Binary Image Classifier

Complete training pipeline using PyTorch:
  1. Dataset loading via torchvision.datasets.ImageFolder
  2. Preprocessing with torchvision.transforms
  3. Custom CNN model definition (PeopleFoodCNN)
  4. Training loop with CrossEntropyLoss + Adam optimizer
  5. Evaluation on validation and test sets each epoch
  6. Saves: model.pth, training_loss.png, test_accuracy.png, training_log.txt

Usage: python3 train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 64          # Resize all images to 64x64
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_CLASSES = 2        # food (0) and people (1)

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Preprocessing with torchvision.transforms
# ============================================================

# Training transforms include augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Validation/test transforms — no augmentation
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ============================================================
# 2. Dataset Loading
# ============================================================

def load_datasets():
    """Load train, val, and test datasets using ImageFolder."""
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "val"),
        transform=eval_transform,
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "test"),
        transform=eval_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Classes: {train_dataset.classes}")
    print(f"Class-to-index mapping: {train_dataset.class_to_idx}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# ============================================================
# 3. Custom CNN Model
# ============================================================

class PeopleFoodCNN(nn.Module):
    """
    Custom Convolutional Neural Network for binary classification.

    Architecture:
        Block 1: Conv2d(3->16, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
        Block 2: Conv2d(16->32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
        Block 3: Conv2d(32->64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
        Flatten -> FC(64*8*8 -> 128) -> ReLU -> Dropout(0.3) -> FC(128 -> 2)

    Input: (batch, 3, 64, 64)
    Output: (batch, 2) — logits for [food, people]
    """

    def __init__(self):
        super(PeopleFoodCNN, self).__init__()

        # Convolutional block 1: 3 -> 16 channels, spatial: 64 -> 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Convolutional block 2: 16 -> 32 channels, spatial: 32 -> 16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Convolutional block 3: 32 -> 64 channels, spatial: 16 -> 8
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Shared layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 3 pooling layers: 64 -> 32 -> 16 -> 8, so feature map is 64 * 8 * 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        # Block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# 4. Training Function
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer):
    """Train the model for one epoch. Returns average training loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


# ============================================================
# 5. Evaluation Function
# ============================================================

def evaluate(model, data_loader):
    """Evaluate model accuracy on a dataset. Returns accuracy as a percentage."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy


# ============================================================
# 6. Visualization & Logging
# ============================================================

def save_plots(train_losses, test_accuracies):
    """Save training loss and test accuracy plots as PNG files."""
    epochs = range(1, len(train_losses) + 1)

    # Training loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "b-o", linewidth=2, markersize=4, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=150)
    plt.close()
    print(f"  Saved: outputs/training_loss.png")

    # Test accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, test_accuracies, "r-o", linewidth=2, markersize=4, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_accuracy.png"), dpi=150)
    plt.close()
    print(f"  Saved: outputs/test_accuracy.png")


def save_training_log(train_losses, val_accuracies, test_accuracies):
    """Save per-epoch training log as a text file."""
    log_path = os.path.join(OUTPUT_DIR, "training_log.txt")
    with open(log_path, "w") as f:
        f.write(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Acc (%)':>12}  {'Test Acc (%)':>12}\n")
        f.write("-" * 50 + "\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1:>6}  {train_losses[i]:>12.4f}  {val_accuracies[i]:>12.2f}  {test_accuracies[i]:>12.2f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Final test accuracy: {test_accuracies[-1]:.2f}%\n")
    print(f"  Saved: outputs/training_log.txt")


def evaluate_per_image(model, class_names):
    """Evaluate each test image individually and report misclassifications by filename."""
    test_dir = os.path.join(DATA_DIR, "test")
    model.eval()
    misclassified = []

    print("Per-image test results:")
    for true_label_name in sorted(os.listdir(test_dir)):
        class_dir = os.path.join(test_dir, true_label_name)
        if not os.path.isdir(class_dir):
            continue
        true_idx = class_names.index(true_label_name)

        for filename in sorted(os.listdir(class_dir)):
            if not filename.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            image = Image.open(os.path.join(class_dir, filename)).convert("RGB")
            tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                _, predicted = torch.max(output, 1)
                pred_idx = predicted.item()

            status = "CORRECT" if pred_idx == true_idx else "WRONG"
            if pred_idx != true_idx:
                misclassified.append((filename, true_label_name, class_names[pred_idx]))
            print(f"  {status:>7}  | True: {true_label_name:<8} | Predicted: {class_names[pred_idx]:<8} | {filename}")

    if misclassified:
        print()
        print("Misclassified images:")
        for fname, true_lbl, pred_lbl in misclassified:
            print(f"  - {fname} (true: {true_lbl}, predicted: {pred_lbl})")

    return misclassified


# ============================================================
# 7. Main — Full Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("People vs Food — CNN Image Classifier (PyTorch)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()

    # Load data
    print("Loading datasets...")
    train_loader, val_loader, test_loader = load_datasets()
    print()

    # Initialize model
    model = PeopleFoodCNN().to(DEVICE)
    print("Model architecture:")
    print(model)
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    train_losses = []
    val_accuracies = []
    test_accuracies = []

    print(f"Training for {NUM_EPOCHS} epochs...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Acc (%)':>12}  {'Test Acc (%)':>12}")
    print("-" * 50)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

        # Evaluate
        val_acc = evaluate(model, val_loader)
        test_acc = evaluate(model, test_loader)

        # Record
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

        print(f"{epoch:>6}  {train_loss:>12.4f}  {val_acc:>12.2f}  {test_acc:>12.2f}")

    print("-" * 50)
    print()

    # Final results
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Best Test Accuracy:  {max(test_accuracies):.2f}% (epoch {test_accuracies.index(max(test_accuracies)) + 1})")
    print()

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  Saved: outputs/model.pth")

    # Save plots and log
    save_plots(train_losses, test_accuracies)
    save_training_log(train_losses, val_accuracies, test_accuracies)

    # Per-image test evaluation
    print()
    class_names = sorted(os.listdir(os.path.join(DATA_DIR, "test")))
    evaluate_per_image(model, class_names)

    print()
    print("Training complete! All outputs saved to outputs/")


if __name__ == "__main__":
    main()
