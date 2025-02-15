import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.utils import save_image
import subprocess

# =============================
# 1. 数据集与预处理
# =============================
data_dir = "/cpfs01/user/hanyujin/causal-dm/Integrated-Design-Diffusion-Model/datasets/sunshadow_lfd_lnd_rfd_rnd/images"
output_dir = "/cpfs01/user/hanyujin/causal-dm/AR_diff/autoregressive-diffusion-pytorch/autoregressive_diffusion_pytorch/reconstruction_images"
save_model_path = "/cpfs01/user/hanyujin/causal-dm/AR_diff/autoregressive-diffusion-pytorch/autoregressive_diffusion_pytorch/pretrained_vae/kl4.pt"
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(32),  # 保证统一大小
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0)  # 归一化到 [-1, 1]
])

# 自定义数据集读取类
class PNGDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        if img.mode == 'RGBA':  # 如果是 4 通道（RGBA）
            img = img.convert('RGB')  # 丢弃 alpha 通道
        elif img.mode != 'RGB':  # 如果是灰度、单通道之类
            img = img.convert('RGB')  # 强制转换成 RGB
        if self.transform:
            img = self.transform(img)
        return img

# 数据加载器
dataset = PNGDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# =============================
# 2. 初始化 AutoencoderKL 模型
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoencoderKL(
    in_channels=3,
    out_channels=3,
    latent_channels=4,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"), 
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),          # 对应增加一层上采样
    block_out_channels=(64, 128, 256)  # 每层输出通道数，调整
)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# =============================
# 3. 训练 AutoencoderKL
# =============================
epochs = 20
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for imgs in dataloader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        recon_imgs = model(imgs).sample  # AutoencoderKL 输出的重建图像
        loss = criterion(recon_imgs, imgs)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    # 保存部分重建结果进行可视化
    model.eval()
    with torch.no_grad():
        sample_imgs = next(iter(dataloader))[:8].to(device)
        recon_imgs = model(sample_imgs).sample
        save_image((recon_imgs + 1) / 2, os.path.join(output_dir, f"reconstructed_epoch_{epoch+1}.png"))
        save_image((sample_imgs + 1) / 2, os.path.join(output_dir, f"original_epoch_{epoch+1}.png"))

# 保存模型
torch.save(model.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")

# =============================
# 4. 计算重建效果 (FID)
# =============================
# # 安装 pytorch-fid
# try:
#     subprocess.run(["pip", "install", "pytorch-fid"], check=True)
# except:
#     print("pytorch-fid 已安装")

from pytorch_fid import fid_score

# 保存重建图像和真实图像的目录
real_images_dir = os.path.join(output_dir, "real_images")
reconstructed_images_dir = os.path.join(output_dir, "reconstructed_images")
os.makedirs(real_images_dir, exist_ok=True)
os.makedirs(reconstructed_images_dir, exist_ok=True)

# 保存真实图像和重建图像
model.eval()
with torch.no_grad():
    for idx, imgs in enumerate(dataloader):
        imgs = imgs.to(device)
        recon_imgs = model(imgs).sample
        
        # 保存图像
        for i in range(imgs.size(0)):
            save_image((imgs[i] + 1) / 2, os.path.join(real_images_dir, f"real_{idx * 8 + i}.png"))
            save_image((recon_imgs[i] + 1) / 2, os.path.join(reconstructed_images_dir, f"reconstructed_{idx * 8 + i}.png"))

# 计算 FID 分数
fid = fid_score.calculate_fid_given_paths([real_images_dir, reconstructed_images_dir], batch_size=8, device=device, dims=2048)
print(f"FID score: {fid:.4f}") #FID score: 0.9543
