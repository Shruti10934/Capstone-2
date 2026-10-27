import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_OUT = os.path.join(BASE_DIR, "half_resnet18.pth")

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3


def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    # Resize images and normalize them using ImageNet statistics
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=transform
    )
    test_set = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, train_set.classes


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, classes = get_dataloaders()
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes.")

    # Load pretrained ResNet-18 and replace its final classification layer
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Evaluate the model without calculating gradients
        model.eval()
        correct, total, val_loss = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_train = train_loss / len(train_loader.dataset)
        avg_val = val_loss / total if total else 0.0
        acc = (correct / total) * 100 if total else 0.0

        print(
            f"Epoch {epoch + 1}: Train Loss = {avg_train:.4f} | "
            f"Val Loss = {avg_val:.4f} | Val Acc = {acc:.2f}%"
        )

        # Save the model whenever validation accuracy improves
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"[Saved Best Model -> {MODEL_OUT}]")

    print(f"\nTraining complete. Highest Validation Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train()
