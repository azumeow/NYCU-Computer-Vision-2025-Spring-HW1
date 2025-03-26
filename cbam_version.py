import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet34_Weights
import pandas as pd


# Configuration
num_classes = 100
batch_size = 64
learning_rate = 0.0005  # from 0.001 to 0.0005
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "data"
OUTPUT_DIR = "ex"


# 通道注意力 (Channel Attention)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        out = self.sigmoid(avg_out + max_out)
        return x * out


# 空間注意力 (Spatial Attention)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


# CBAM 模塊 (Channel + Spatial)
class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# Transform
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                           hue=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),  # from 232 to 256
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder('train', transform=train_transform)
val_dataset = datasets.ImageFolder('val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4)

idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}


# Model initialization
model = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
# 在 layer2、layer3、layer4 加入 CBAM
model.layer2 = nn.Sequential(
    model.layer2,
    CBAMBlock(128)  # layer2 的輸出通道數為 128
)
model.layer3 = nn.Sequential(
    model.layer3,
    CBAMBlock(256)  # layer3 的輸出通道數為 256
)
model.layer4 = nn.Sequential(
    model.layer4,
    CBAMBlock(512)  # layer4 的輸出通道數為 512
)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Training loop
def train_model():
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, '
              f'Accuracy: {train_acc:.2f}%')

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        scheduler.step()

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_resnet34_model.pth')


def test_model():
    model.load_state_dict(torch.load('best_resnet34_model.pth'))
    model.eval()

    test_images = sorted(os.listdir(os.path.join(DATA_DIR, 'test')))
    data = []

    with torch.no_grad():
        for img_name in test_images:
            img_path = os.path.join(DATA_DIR, 'test', img_name)
            image = datasets.folder.default_loader(img_path)
            image = val_transform(image).unsqueeze(0).to(device)
            output = model(image)
            _, predicted = output.max(1)
            predicted_class = idx_to_class[predicted.item()]
            image_name = os.path.splitext(img_name)[0]
            data.append([image_name, predicted_class])

    # Save CSV
    pd.DataFrame(data, columns=['image_name', 'pred_label']) \
        .to_csv(os.path.join(OUTPUT_DIR, 'prediction.csv'), index=False)


# Run training and testing
if __name__ == '__main__':
    train_model()
    test_model()
