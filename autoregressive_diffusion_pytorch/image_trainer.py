from typing import List

import math
from pathlib import Path
import wandb
from accelerate import Accelerator
from ema_pytorch import EMA
import torchvision.transforms.functional as F
import torch
from torch import nn
from torch.optim import Adam, AdamW
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

# dataset classes

# class ImageDataset(Dataset):
#     def __init__(
#         self,
#         folder: str | Path,
#         image_size: int,
#         exts: List[str] = ['jpg', 'jpeg', 'png', 'tiff'],
#         augment_horizontal_flip = False,
#         convert_image_to = None
#     ):
#         super().__init__()
#         if isinstance(folder, str):
#             folder = Path(folder)

#         assert folder.is_dir()

#         self.folder = folder
#         self.image_size = image_size

#         self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

#         def convert_image_to_fn(img_type, image):
#             if image.mode == img_type:
#                 return image

#             return image.convert(img_type)

#         maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

#         self.transform = T.Compose([
#             T.Lambda(maybe_convert_fn),
#             T.Resize(image_size),
#             T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
#             T.CenterCrop(image_size),
#             T.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.paths)

#     # def __getitem__(self, index):
#     #     path = self.paths[index]
#     #     img = Image.open(path)
#     #     return self.transform(img)

#     def __getitem__(self, index):
#        path = self.paths[index]
#        img = Image.open(path)
#        transformed_img = self.transform(img)
#     #    print(f"Image shape: {transformed_img.shape}")  # 添加这行
#        return transformed_img

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
    def __init__(self, model, *, dataset: Dataset, dataset_name, patch_size,image_size, dim,device,no_weight,depth, mlp_depth, num_train_steps=70_000, learning_rate=3e-4, batch_size=16, 
                 checkpoints_folder='./checkpoints', results_folder='./results', save_results_every=100, 
                 checkpoint_every=1000, num_samples=16, adam_kwargs=dict(), accelerate_kwargs=dict(), ema_kwargs=dict(),use_parallel_order = False,reverse=False):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)
        self.model = model
        self.device = device
        if self.is_main:
            self.ema_model = EMA(
                self.model,
                forward_method_names=('sample',),
                **ema_kwargs
            )
            self.ema_model.to(self.device)#self.accelerator.device)

        self.optimizer = Adam(model.parameters(), lr=learning_rate, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)
        self.num_train_steps = num_train_steps
        self.no_weight = no_weight
        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)
        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every
        self.reverse = reverse
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples
        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

        # Initialize wandb
        wandb.init(
            project="autoregressive-diffusion",
            name=f"ar_{dataset_name}_numstep_{num_train_steps}_parallel{use_parallel_order}_patch_size{patch_size}_imagesize{image_size}_dim{dim}_noweightedloss{self.no_weight}_depthmlpdepth{depth}{mlp_depth}",
            config={
                "num_train_steps": num_train_steps,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "use_parallel_order": use_parallel_order
            }
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )

        torch.save(save_package, str(self.checkpoints_folder / path))


    def forward(self):
        dl = cycle(self.dl)
        for ind in range(self.num_train_steps):
            step = ind + 1
            self.model.to(self.device).train()
            data = next(dl).to(self.device)
            diffusion_loss, diffusion_loss_noweight, latent_min, latent_max = self.model(data)
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
                K = (self.image_size / self.patch_size)**2
                wandb.log({
                    "loss": loss.item()* K,
                    "single_loss": loss.item()
                }, step=step)

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if divisible_by(step, self.save_results_every):
                    with torch.no_grad():
                        sampled = self.ema_model.sample(batch_size=self.num_samples, latent_min=latent_min, latent_max=latent_max)
                    
                    if self.reverse:
                        sampled = F.vflip(sampled)  #

                    save_image(sampled, str(self.results_folder / f'results.{step}.png'), nrow=self.num_sample_rows)
                    wandb.log({"generated_samples": [wandb.Image(sampled, caption=f"Step {step}")]}, step=step)  # Log generated samples

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
