import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import os


# 1. Lightweight CNN
class LightweightCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), p=2, dim=1)


# 2. CLIP-like model with learnable class embeddings
class FireClipModel(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=2):
        super().__init__()
        self.image_encoder = LightweightCNN(embedding_dim)
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, embedding_dim))

    def forward(self, x):
        image_emb = self.image_encoder(x)
        class_emb = F.normalize(self.class_embeddings, p=2, dim=1)
        return torch.matmul(image_emb, class_emb.T)

# 3. Load and split data
def load_and_split_data(data_dir, batch_size=64, val_ratio=0.2):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    total_len = len(full_dataset)
    val_len = int(val_ratio * total_len)
    train_len = total_len - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, full_dataset.classes

# 4. Train function
def train(model, loader, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(100):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# 5. Evaluation function with classification report
def evaluate(model, loader, device, class_names):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predicted = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(predicted)
            trues.extend(labels.numpy())

    report = classification_report(trues, preds, target_names=class_names)
    print(report)

    # Save report to file
    with open("cnn_clip_scalogram_kaggle.txt", "w") as f:
        f.write(report)
    print("✅ Classification report saved as 'cnn_clip_scalogram_kagglecompressed.txt")


# 6. Main
if __name__ == "__main__":
    dataset_path = r"C:\Users\addmin\Downloads\cwt_scalogram\scalogram_kaggel_compression"  # Directory with subfolders: fire/, nonfire/
    train_loader, val_loader, class_names = load_and_split_data(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FireClipModel()

    train(model, train_loader, device)
    evaluate(model, val_loader, device, class_names)

    torch.save(model.state_dict(), "cnn_clip_scalogram_kaggle_compr.pth")
    print("✅ Model saved as fire_clip_kaggle.pth")