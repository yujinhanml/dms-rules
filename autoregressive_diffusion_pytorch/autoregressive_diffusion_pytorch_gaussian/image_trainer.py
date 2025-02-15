from typing import List
import matplotlib.pyplot as plt
import math
from pathlib import Path
import wandb
from accelerate import Accelerator
from ema_pytorch import EMA
import torchvision.transforms.functional as F
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as T

from PIL import Image

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(dl):
    while True:
        for batch in dl:
            yield batch


class ImageDataset(Dataset):
    def __init__(
        self,
        folder: str | Path,
        image_size: int,
        exts: List[str] = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None,
        reverse=False 
    ):
        super().__init__()
        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.is_dir()

        self.folder = folder
        self.image_size = image_size
        self.reverse = reverse

        # 拍平所有符合扩展文件类型的文件路径
        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        def convert_image_to_fn(img_type, image):
            if image.mode == img_type:
                return image
            return image.convert(img_type)

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # 转换 pipeline
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # 获取图像路径
        path = self.paths[index]
        img = Image.open(path)

        # 检查并转换通道类型（比如 RGBA -> RGB 或 L -> RGB）
        if img.mode == 'RGBA':  # 如果是 4 通道（RGBA）
            img = img.convert('RGB')  # 丢弃 alpha 通道
        elif img.mode != 'RGB':  # 如果是灰度、单通道之类
            img = img.convert('RGB')  # 强制转换成 RGB
        if self.reverse:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # 应用其他数据增强和处理
        transformed_img = self.transform(img)

        return transformed_img



# trainer

class ImageTrainer(Module):
    def __init__(self, model, *, dataset: Dataset, dim, device, no_weight, dataset_name,patch_size,num_train_steps=70_000,
                 learning_rate=3e-4, batch_size=16, checkpoints_folder='./checkpoints',
                 results_folder='./results', save_results_every=100, checkpoint_every=1000, num_samples=16,
                 adam_kwargs=dict(), accelerate_kwargs=dict(), ema_kwargs=dict()):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)
        self.model = model
        self.device = device
        self.no_weight = no_weight

        if self.is_main:
            self.ema_model = EMA(
                self.model,
                forward_method_names=('sample',),
                **ema_kwargs
            )
            self.ema_model.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=learning_rate, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)
        self.num_train_steps = num_train_steps
        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)
        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        self.save_results_every = save_results_every
        self.checkpoint_every = checkpoint_every
        self.num_samples = batch_size

        # Initialize wandb
        wandb.init(
            project="autoregressive-diffusion",
            name=f"ar_{dataset_name}_numstep_{num_train_steps}_dim{dim}_noweightedloss{self.no_weight}_patchsize{patch_size}",
            config={
                "num_train_steps": num_train_steps,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model=self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model=self.ema_model.state_dict(),
            optimizer=self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )
        torch.save(save_package, str(self.checkpoints_folder / path))

    def forward(self):
        dl = cycle(self.dl)
        for ind in range(self.num_train_steps):
            step = ind + 1
            self.model.to(self.device).train()
            data = next(dl).to(self.device)
            diffusion_loss, diffusion_loss_noweight = self.model(data)

            if self.no_weight:
                loss = diffusion_loss_noweight
            else:
                loss = diffusion_loss

            self.accelerator.print(f'[{step}] loss: {loss.item():.3f}')
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main:
                self.ema_model.update()
                wandb.log({"loss": loss.item()}, step=step)

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if divisible_by(step, self.save_results_every):
                    with torch.no_grad():
                        sampled = self.ema_model.sample(batch_size=self.num_samples)

                    # 保存生成的结果为 CSV
                    sampled = sampled.squeeze(-1)
                    save_path = self.results_folder / f'sampled_results.{step}.csv'
                    np.savetxt(save_path, sampled.cpu().numpy(), delimiter=',', header='x1,x2', comments='')
                    # wandb.log({"sampled_results_path": str(save_path)}, step=step)

                    sampled_np = sampled.cpu().numpy()  # 转为 numpy 数组
                    plt.scatter(sampled_np[:, 0], sampled_np[:, 1], alpha=0.7)
                    plt.title(f'Scatter Plot at Step {step}')
                    plt.xlabel('x1')
                    plt.ylabel('x2')
                    scatter_path = self.results_folder / f'scatter_plot.{step}.png'
                    plt.savefig(scatter_path)
                    plt.close()

                if divisible_by(step, self.checkpoint_every):
                    self.save(f'checkpoint.{step}.pt')

            self.accelerator.wait_for_everyone()

        print('training complete')


# class ImageTrainer(Module):
#     def __init__(
#         self,
#         model,
#         *,
#         dataset: Dataset,
#         num_train_steps = 70_000,
#         learning_rate = 3e-4,
#         batch_size = 16,
#         checkpoints_folder: str = './checkpoints',
#         results_folder: str = './results',
#         save_results_every: int = 100,
#         checkpoint_every: int = 1000,
#         num_samples: int = 16,
#         adam_kwargs: dict = dict(),
#         accelerate_kwargs: dict = dict(),
#         ema_kwargs: dict = dict(),
#         project_name="AR_diffusion"
#     ):
#         super().__init__()
#         self.accelerator = Accelerator(**accelerate_kwargs)

#         self.model = model

#         if self.is_main:
#             self.ema_model = EMA(
#                 self.model,
#                 forward_method_names = ('sample',),
#                 **ema_kwargs
#             )

#             self.ema_model.to(self.accelerator.device)

#             wandb.init(
#                 project=project_name,
#                 config={
#                     "num_train_steps": num_train_steps,
#                     "learning_rate": learning_rate,
#                     "batch_size": batch_size,
#                     "image_size": dataset.image_size,
#                     "use_parallel_order": model.use_parallel_order
#                 }
#             )
#             wandb.watch(self.model, log="all")


#         self.optimizer = Adam(model.parameters(), lr = learning_rate, **adam_kwargs)
#         self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

#         self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

#         self.num_train_steps = num_train_steps

#         self.checkpoints_folder = Path(checkpoints_folder)
#         self.results_folder = Path(results_folder)

#         self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
#         self.results_folder.mkdir(exist_ok = True, parents = True)

#         self.checkpoint_every = checkpoint_every
#         self.save_results_every = save_results_every

#         self.num_sample_rows = int(math.sqrt(num_samples))
#         assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
#         self.num_samples = num_samples

#         assert self.checkpoints_folder.is_dir()
#         assert self.results_folder.is_dir()

#     @property
#     def is_main(self):
#         return self.accelerator.is_main_process

#     def save(self, path):
#         if not self.is_main:
#             return

#         save_package = dict(
#             model = self.accelerator.unwrap_model(self.model).state_dict(),
#             ema_model = self.ema_model.state_dict(),
#             optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
#         )

#         torch.save(save_package, str(self.checkpoints_folder / path))

#     def forward(self):

#         dl = cycle(self.dl)

#         for ind in range(self.num_train_steps):
#             step = ind + 1

#             self.model.train()

#             data = next(dl)
#             loss, latent_min ,latent_max = self.model(data)

#             self.accelerator.print(f'[{step}] loss: {loss.item():.3f}')
#             self.accelerator.backward(loss)

#             self.optimizer.step()
#             self.optimizer.zero_grad()

#             if self.is_main:
#                 self.ema_model.update()
#                 wandb.log({"loss": loss.item()}, step=step)
                

#             self.accelerator.wait_for_everyone()

#             if self.is_main:
#                 if divisible_by(step, self.save_results_every):

#                     with torch.no_grad():
#                         sampled = self.ema_model.sample(batch_size = self.num_samples,latent_min = latent_min,latent_max=latent_max)

#                     # sampled.clamp_(0., 1.)
#                     save_image(sampled, str(self.results_folder / f'results.{step}.png'), nrow = self.num_sample_rows)
#                     wandb.log({"generated_samples": [wandb.Image(sampled, caption=f"Step {step}")]}, step=step)

#                 if divisible_by(step, self.checkpoint_every):
#                     self.save(f'checkpoint.{step}.pt')

#             self.accelerator.wait_for_everyone()


#         print('training complete')
