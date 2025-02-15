import os
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

# =============================
# 1. 数据集与预处理
# =============================
data_dir = "/cpfs01/user/hanyujin/causal-dm/Integrated-Design-Diffusion-Model/datasets/sunshadow_lfd_lnd_rfd_rnd/images"
output_dir = "/cpfs01/user/hanyujin/causal-dm/AR_diff/autoregressive-diffusion-pytorch/autoregressive_diffusion_pytorch/reconstruction_images_vq_vae"
save_model_path = "/cpfs01/user/hanyujin/causal-dm/AR_diff/autoregressive-diffusion-pytorch/autoregressive_diffusion_pytorch/pretrained_vae/vqvae.pt"
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
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# 数据加载器
dataset = PNGDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# =============================
# 2. VQ-VAE 模型定义
# =============================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        z_flattened = z.view(-1, self.embedding_dim)
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) 
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.t()) 
                     + torch.sum(self.embedding.weight ** 2, dim=1))
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view_as(z)
        
        # 损失
        commitment_loss = self.beta * torch.mean((quantized.detach() - z) ** 2)
        codebook_loss = torch.mean((quantized - z.detach()) ** 2)
        loss = commitment_loss + codebook_loss

        quantized = z + (quantized - z).detach()
        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=64, num_embeddings=512):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, embedding_dim, kernel_size=3, stride=1, padding=1)
        )

        # 向量量化模块
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.quantizer(z)
        recon_x = self.decoder(quantized)
        return recon_x, vq_loss

# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VQVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# =============================
# 3. 训练 VQ-VAE
# =============================
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for imgs in dataloader:
        imgs = imgs.to(device)
        optimizer.zero_grad()

        # 前向传播
        recon_imgs, vq_loss = model(imgs)
        recon_loss = criterion(recon_imgs, imgs)
        loss = recon_loss + vq_loss

        # 反向传播与优化
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    # 保存部分重建结果进行可视化
    model.eval()
    with torch.no_grad():
        sample_imgs = next(iter(dataloader))[:8].to(device)
        recon_imgs, _ = model(sample_imgs)
        save_image((recon_imgs + 1) / 2, os.path.join(output_dir, f"reconstructed_epoch_{epoch+1}.png"))
        save_image((sample_imgs + 1) / 2, os.path.join(output_dir, f"original_epoch_{epoch+1}.png"))

# 保存模型
torch.save(model.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")

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
        recon_imgs,_ = model(imgs)
        
        # 保存图像
        for i in range(imgs.size(0)):
            save_image((imgs[i] + 1) / 2, os.path.join(real_images_dir, f"real_{idx * 8 + i}.png"))
            save_image((recon_imgs[i] + 1) / 2, os.path.join(reconstructed_images_dir, f"reconstructed_{idx * 8 + i}.png"))

# 计算 FID 分数
fid = fid_score.calculate_fid_given_paths([real_images_dir, reconstructed_images_dir], batch_size=8, device=device, dims=2048)
print(f"FID score: {fid:.4f}") #FID score: 0.9543
