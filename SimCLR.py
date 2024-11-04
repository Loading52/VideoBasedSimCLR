import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from contrastive_loss import NTXentLoss  # 导入对比损失函数


# 定义SimCLR模型，包含编码器和投影头
class SimCLRModel(nn.Module):
    def __init__(self, encoder_arch='resnet18', feature_dim=128):
        super(SimCLRModel, self).__init__()
        # 初始化编码器
        self.encoder = getattr(models, encoder_arch)(pretrained=False)
        self.encoder.fc = nn.Identity()  # 去掉分类头，保留特征提取

        # 定义投影头
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.projection_head(features)


# 图像数据增强封装
def get_data_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(100),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.ToTensor()
    ])


# 训练过程封装
class SimCLRTrainer:
    def __init__(self, model, temperature=0.5, lr=1e-3):
        self.model = model
        self.criterion = NTXentLoss(temperature=temperature)  # SimCLR对比损失
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataloader, num_epochs, device):
        self.model.to(device)
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in dataloader:  # dataloader包含成对的正样本(img1, img2)
                img1, img2 = data
                img1, img2 = get_data_transforms()(img1).to(device), get_data_transforms()(img2).to(device)

                # 前向传播
                z1, z2 = self.model(img1), self.model(img2)

                # 合并批次，包含成对的样本
                representations = torch.cat([z1, z2], dim=0)

                # 计算损失
                loss = self.criterion(representations)
                epoch_loss += loss.item()

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


# 使用示例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLRModel(encoder_arch='resnet18', feature_dim=128)
trainer = SimCLRTrainer(model, temperature=0.5, lr=1e-3)

# 假设dataloader是一个PyTorch DataLoader，包含成对的样本
# dataloader = ...

# 开始训练
# trainer.train(dataloader, num_epochs=10, device=device)
