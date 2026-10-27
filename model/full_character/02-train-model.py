import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(data_dir, output_path, epochs, batch_size, lr):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if not os.path.exists(train_dir):
        print(f"Error: Training directory {train_dir} not found.")
        return

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    num_classes = len(train_dataset.classes)
    print(f"Detected {num_classes} classes.")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct / total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(
            f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}"
        )

    print("\nGenerating Classification Report...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    class_names = train_dataset.classes
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=90)
    plt.title("Confusion Matrix")
    cm_path = os.path.join(os.path.dirname(output_path), "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Plot Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    curves_path = os.path.join(os.path.dirname(output_path), "training_curves.png")
    plt.savefig(curves_path)
    print(f"Training curves saved to {curves_path}")

    print(f"Saving model to {output_path}")
    torch.save(model.state_dict(), output_path)
    print("Training complete.")


if __name__ == "__main__":
    DATA_DIR = "model/full_character/dataset"
    OUTPUT_PATH = "model/full_character/resnet18.pth"
    BATCH_SIZE = 64
    EPOCHS = 11
    LR = 0.001

    train_model(DATA_DIR, OUTPUT_PATH, EPOCHS, BATCH_SIZE, LR)
