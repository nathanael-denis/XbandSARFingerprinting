import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import random
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import pandas as pd

'''



'''

job_id = os.getenv("SLURM_JOB_ID", "default_job")
os.makedirs("results", exist_ok=True)
csv_path = os.path.join("results", f"confusion_matrix_{job_id}.csv")

# ---------- Optional AWGN Transform ----------
class AddAWGN(object):
    def __init__(self, snr_db_range=(-5, 15)):
        self.snr_db_range = snr_db_range

    def __call__(self, tensor):
        snr_db = random.uniform(*self.snr_db_range)
        snr_linear = 10 ** (snr_db / 10)
        signal_power = tensor.pow(2).mean()
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(tensor) * noise_power.sqrt()
        return tensor + noise

# ---------- Supervised Contrastive Transform ----------
class SupervisedContrastiveTransform:
    def __init__(self, mean=0.485, std=0.229):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
            AddAWGN(snr_db_range=(10, 30))
        ])

    def __call__(self, x):
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj

# ---------- Dataset for Contrastive Training ----------
class SupervisedContrastiveDataset(Dataset):
    def __init__(self, root_dir, mean=0.485, std=0.229):
        self.dataset = datasets.ImageFolder(root=root_dir, loader=self.pil_loader_grayscale)
        self.transform = SupervisedContrastiveTransform(mean, std)

    @staticmethod
    def pil_loader_grayscale(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = self.dataset.loader(path)
        xi, xj = self.transform(image)
        return xi, xj, label

# ---------- Dataset for Classification ----------
class ClassificationDataset(Dataset):
    def __init__(self, root_dir, mean=0.485, std=0.229):
        self.dataset = datasets.ImageFolder(root=root_dir, loader=self.pil_loader_grayscale)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ])

    @staticmethod
    def pil_loader_grayscale(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = self.dataset.loader(path)
        image = self.transform(image)
        return image, label

# ---------- Model (Modified to handle a single channel input with EfficientNet-B0) ----------
class SupConEfficientNet(nn.Module):
    def __init__(self, base_model='efficientnet_b0', projection_dim=128):
        super().__init__()
        self.encoder = getattr(models, base_model)(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        original_conv_stem = self.encoder.features[0][0]
        self.encoder.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv_stem.out_channels,
            kernel_size=original_conv_stem.kernel_size,
            stride=original_conv_stem.stride,
            padding=original_conv_stem.padding,
            bias=original_conv_stem.bias
        )
        
        with torch.no_grad():
            self.encoder.features[0][0].weight.copy_(
                torch.sum(original_conv_stem.weight, dim=1, keepdim=True)
            )

        num_ftrs = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(),
            nn.Linear(num_ftrs, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)

# ---------- Supervised Contrastive Loss ----------
def supervised_contrastive_loss(features, labels, temperature=0.07):
    device = features.device
    batch_size = features.size(0)

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    contrast = torch.matmul(features, features.T) / temperature
    logits_max, _ = contrast.max(dim=1, keepdim=True)
    logits = contrast - logits_max.detach()

    logits_mask = torch.ones_like(mask).fill_diagonal_(0)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

    loss = -mean_log_prob_pos.mean()
    return loss

# ---------- Classification Head ----------
class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_ftrs, num_classes):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.fc(features)

# ---------- Directories ----------
base_dir = "output"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# ---------- Compute Dataset Statistics ----------
def compute_dataset_stats(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    mean, std = 0, 0
    for images, _ in loader:
        batch_mean = images.mean(dim=[0, 2, 3])
        batch_std = images.std(dim=[0, 2, 3])
        mean += batch_mean
        std += batch_std
    mean /= len(loader)
    std /= len(loader)
    return mean.item(), std.item()

train_dataset = ClassificationDataset(train_dir)
mean, std = compute_dataset_stats(train_dataset)
print(f"Dataset mean: {mean:.4f}, std: {std:.4f}")

# ---------- Dataloaders ----------
train_dataset = SupervisedContrastiveDataset(train_dir, mean=mean, std=std)
val_dataset = SupervisedContrastiveDataset(val_dir, mean=mean, std=std)
test_dataset = ClassificationDataset(test_dir, mean=mean, std=std)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Get satellite names from dataset
satellite_names = train_dataset.dataset.classes

# ---------- Train SupCon ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SupConEfficientNet(base_model='efficientnet_b0').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def train_supcon(model, loader, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xi, xj, labels in loader:
            xi, xj, labels = xi.to(device), xj.to(device), labels.to(device)
            features = torch.cat([model(xi), model(xj)], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            loss = supervised_contrastive_loss(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - SupCon Loss: {total_loss / len(loader):.4f}")

train_supcon(model, train_loader, optimizer, epochs=40)
torch.save(model.encoder.state_dict(), "supcon_encoder.pth")

# ---------- Classification Phase ----------
encoder = getattr(models, 'efficientnet_b0')(weights=None)
original_conv_stem = encoder.features[0][0]
encoder.features[0][0] = nn.Conv2d(
    in_channels=1,
    out_channels=original_conv_stem.out_channels,
    kernel_size=original_conv_stem.kernel_size,
    stride=original_conv_stem.stride,
    padding=original_conv_stem.padding,
    bias=original_conv_stem.bias
)
num_ftrs = encoder.classifier[1].in_features
encoder.classifier = nn.Identity()
encoder.load_state_dict(torch.load("supcon_encoder.pth"))

train_class_dataset = ClassificationDataset(train_dir, mean=mean, std=std)
val_class_dataset = ClassificationDataset(val_dir, mean=mean, std=std)
train_class_loader = DataLoader(train_class_dataset, batch_size=32, shuffle=True)
val_class_loader = DataLoader(val_class_dataset, batch_size=32, shuffle=False)

num_classes = len(train_class_dataset.dataset.classes)
class_counts = [sum(1 for target in train_class_dataset.dataset.targets if target == i) for i in range(num_classes)]
print("Class counts:", class_counts)  # Debug print
class_weights = torch.tensor([1.0 / (count + 1e-6) for count in class_counts]).to(device)
classifier = LinearClassifier(encoder, num_ftrs, num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(classifier.fc.parameters(), lr=1e-3)
# ---------- Train Classifier ----------
def train_classifier(model, loader, optimizer, criterion, epochs=10):
    model.train()
    best_f1 = 0
    for epoch in range(epochs):
        total_loss = 0
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.4f}")
        f1, _ = evaluate(classifier, val_class_loader)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(classifier.state_dict(), "best_classifier.pth")

# ---------- Evaluate ----------
def evaluate(model, loader, satellite_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"F1 Score: {f1:.4f}")
    
    # Save confusion matrix with satellite names if provided
    if satellite_names is not None:
        df_cm = pd.DataFrame(cm, index=satellite_names, columns=satellite_names)
        df_cm.to_csv(csv_path)
    return f1, cm

# ---------- Run Classification ----------
train_classifier(classifier, train_class_loader, optimizer, criterion, epochs=20)
print("Validation Performance:")
evaluate(classifier, val_class_loader, satellite_names)
print("Test Performance:")
evaluate(classifier, test_loader, satellite_names)
