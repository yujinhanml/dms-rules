import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import trange
import wandb
import argparse
import torch.nn.functional as F
# 定义 ResNet-8 模型
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet8(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet8, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = BasicBlock(16, 16, stride=1)
        self.layer2 = BasicBlock(16, 32, stride=2)
        self.layer3 = BasicBlock(32, 64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, num_classes=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleMLP(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, num_classes=3):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = (self.fc1(x))
        x = (self.fc2(x))
        return x

# 定义 U-Net 模型
class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.global_pool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展平
        return x
# 初始化 wandb
def init_wandb(model_name, num_epochs, batch_size, learning_rate,folder_name,noised):
    wandb.init(
        project="guided-diffusion",
        name=f"train_{model_name}_{num_epochs}_{folder_name}_noised{noised}",
        config={
            "architecture": model_name,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
    )

# 模型选择函数
def select_model(model_name, num_classes):
    if model_name == "MLP":
        return MLP(num_classes=num_classes).to(device)
    elif model_name == "SimpleMLP":
        return SimpleMLP(num_classes=num_classes).to(device)
    elif model_name == "ResNet":
        return ResNet8(num_classes=num_classes).to(device)
    elif model_name == "UNet":
        return UNet(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# 训练和测试函数
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # print("labels:",labels)
        # print("predicted:",predicted)
    epoch_loss = running_loss / total
    epoch_accuracy = 100. * correct / total
    # print("epoch_accuracy:",epoch_accuracy)
    return epoch_loss, epoch_accuracy

def test_model(model, test_loader, criterion, device, num_classes=3):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes  # 每个类别的正确预测数
    class_total = [0] * num_classes    # 每个类别的总样本数

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 统计每个类别的正确预测数和总样本数
            for i in range(num_classes):
                class_mask = (labels == i)
                class_correct[i] += (predicted[class_mask] == labels[class_mask]).sum().item()
                class_total[i] += class_mask.sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100. * correct / total

    # 计算每个类别的准确率
    class_accuracy = [100. * class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(num_classes)]

    return epoch_loss, epoch_accuracy, class_accuracy

# 主函数
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--model_name", type=str, default="MLP", choices=["MLP", "ResNet", "UNet", "SimpleMLP"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="/cpfs04/user/hanyujin/causal-dm/Integrated-Design-Diffusion-Model/datasets/sunshadow_contrastive_ternary")
    parser.add_argument("--save_dir", type=str, default="/cpfs04/user/hanyujin/causal-dm/contrastive_dp/models")
    parser.add_argument("--noised", action="store_true", help="Apply DDPM t=1 noise to images")
    args = parser.parse_args()

    # 设备设置
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DDPM 噪声参数
    beta_1 = 0.0001  # 你可以根据 DDPM 预定义的 noise schedule 选择适当的值
    alpha_1 = 1 - beta_1
    bar_alpha_1 = alpha_1  # 在 t=1 时，累计噪声参数等于 alpha_1
    parts = args.data_path.split("/")
    args.folder_name = parts[-1].replace('"', '')

    # 定义添加 t=1 噪声的函数
    def add_ddpm_noise(img, t=1.0, noise=None, bar_alpha_1=bar_alpha_1):
        # 添加噪声前确保输入是 [batch_size, 3, 32, 32]
        # print("img:",img.shape)
        if noise is None:
            noise = torch.randn_like(img)
        bar_alpha_1 = torch.tensor(bar_alpha_1).to(img.device) 
        # 根据DDPM噪声方程添加噪声
        noisy_img = torch.sqrt(bar_alpha_1) * img + torch.sqrt(1 - bar_alpha_1) * noise
        noisy_img = F.interpolate(noisy_img.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)
        return noisy_img


    # 定义数据变换
    transform_list = [
    transforms.ToTensor(),
    transforms.Lambda(lambda img: add_ddpm_noise(img, noise=torch.randn_like(img), bar_alpha_1=bar_alpha_1))]

    # 组合转换
    transform = transforms.Compose(transform_list)
   

    # 加载数据
    dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    print(f"Dataset loaded. Noised: {args.noised}")


    # 模型初始化
    model = select_model(args.model_name, num_classes=3)
    init_wandb(args.model_name, args.num_epochs, args.batch_size, args.learning_rate,args.folder_name,args.noised)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 创建保存模型的目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    for epoch in trange(1, args.num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_accuracy,class_accuracy = test_model(model, test_loader, criterion, device)

        # Log 到 wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_class1_accuracy": class_accuracy[1],
            "test_class0_accuracy": class_accuracy[0],
            "test_class2_accuracy": class_accuracy[2]
        })

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{args.num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
            for i in range(3):
                print(f"Class {i} Accuracy: {class_accuracy[i]:.2f}%")

        # 保存模型
        if epoch == args.num_epochs:
            model_path = os.path.join(args.save_dir, f"{args.folder_name}_{args.model_name}_{epoch}_noised{args.noised}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    print("Training complete!")
    wandb.finish()