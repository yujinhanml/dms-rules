#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/23 22:26
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from model.networks.base import BaseNet
from model.modules.attention import SelfAttention
from model.modules.block import DownBlock, UpBlock
from model.modules.conv import DoubleConv
import torch
import torch.nn as nn
import torch.nn.functional as F

# one block
# class UNet(BaseNet):
#     """
#     Modified UNet with only one downsampling and one upsampling
#     """

#     def __init__(self, **kwargs):
#         """
#         Initialize the modified UNet
#         """
#         super(UNet, self).__init__(**kwargs)

#         # Initial convolution (no size change)
#         self.inc = DoubleConv(in_channels=self.in_channel, out_channels=self.channel[1], act=self.act)

#         # Single downsampling block (halves spatial size)
#         self.down1 = DownBlock(in_channels=self.channel[1], out_channels=self.channel[2], act=self.act)
#         # self.sa1 = SelfAttention(channels=self.channel[2], size=self.image_size_list[1], act=self.act)

#         # Intermediate processing blocks (maintains size)
#         self.bot1 = DoubleConv(in_channels=self.channel[2], out_channels=self.channel[2], act=self.act)
#         self.bot2 = DoubleConv(in_channels=self.channel[2], out_channels=self.channel[2], act=self.act)
#         self.bot3 = DoubleConv(in_channels=self.channel[2], out_channels=self.channel[2], act=self.act)

#         # Single upsampling block (restores original size)
#         # Input channels: intermediate features + skip connection
#         self.up1 = UpBlock(
#             in_channels=self.channel[2] + self.channel[1],  # 合并中间特征和跳跃连接
#             out_channels=self.channel[1], 
#             act=self.act
#         )
#         # self.sa4 = SelfAttention(channels=self.channel[1], size=self.image_size_list[0], act=self.act)

#         # Final output convolution
#         self.outc = nn.Conv2d(in_channels=self.channel[1], out_channels=self.out_channel, kernel_size=1)

#     def forward(self, x, time, y=None):
#         # Time embedding processing
#         time = time.unsqueeze(-1).type(torch.float)
#         time = self.pos_encoding(time, self.time_channel)
#         if y is not None:
#             time += self.label_emb(y)

#         # Encoder path
#         x1 = self.inc(x)                   # 初始特征 [N, C1, H, W]
#         x2 = self.down1(x1, time)          # 下采样 [N, C2, H/2, W/2]
#         x2_sa = x2        # 自注意力 [N, C2, H/2, W/2]

#         # Intermediate processing
#         bot_out = self.bot3(self.bot2(self.bot1(x2_sa)))  # [N, C2, H/2, W/2]

#         # Decoder path with skip connection
#         up1_out = self.up1(bot_out, x1, time)  # 上采样并拼接 [N, C1, H, W]
#         up1_sa = up1_out            # 最终自注意力

#         return self.outc(up1_sa)

# CNN
# class UNet(BaseNet):
#     """
#     UNet (Simplified: Using Only DoubleConv)
#     """

#     def __init__(self, **kwargs):
#         super(UNet, self).__init__(**kwargs)

#         # **转换 act 为 PyTorch 激活函数**
#         if isinstance(self.act, str):  
#             self.act = self._get_activation(self.act)

#         # **时间 MLP 处理**
#         self.time_embed = nn.Linear(1, self.time_channel)  

#         # Time MLPs (修改输出通道数)
#         self.time_mlp_64 = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(self.time_channel, self.channel[1])  
#         )
#         self.time_mlp_128 = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(self.time_channel, self.channel[2])  
#         )
#         self.time_mlp_256 = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(self.time_channel, self.channel[3])  
#         )

#         # **网络结构**
#         self.inc = DoubleConv(in_channels=self.in_channel, out_channels=self.channel[1], act=self.act)
#         self.down1 = DoubleConv(in_channels=self.channel[1], out_channels=self.channel[2], act=self.act)
#         self.down2 = DoubleConv(in_channels=self.channel[2], out_channels=self.channel[3], act=self.act)

#         self.bot1 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)
#         self.bot2 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)
#         self.bot3 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)

#         # 这里的 `in_channels` 需要加上 skip_x 的通道数
#         self.up2 = DoubleConv(in_channels=self.channel[3] + self.channel[2], out_channels=self.channel[2], act=self.act)
#         self.up3 = DoubleConv(in_channels=self.channel[2] + self.channel[1], out_channels=self.channel[1], act=self.act)

#         self.outc = nn.Conv2d(in_channels=self.channel[1], out_channels=self.out_channel, kernel_size=1)

#     def _get_activation(self, act_str):
#         """把字符串转换为 PyTorch 激活函数"""
#         act_dict = {
#             "relu": nn.ReLU(),
#             "leaky_relu": nn.LeakyReLU(),
#             "gelu": nn.GELU(),
#             "silu": nn.SiLU(),
#             "elu": nn.ELU(),
#         }
#         return act_dict.get(act_str.lower(), nn.ReLU())  # 默认 ReLU

#     def forward(self, x, time, y=None):
#         # 1. 处理时间嵌入
#         time = time.unsqueeze(-1).to(torch.float)  # (batch, 1)
#         time = self.time_embed(time)  # (batch, time_channel)
        
#         time_embed_64 = self.time_mlp_64(time)[:, :, None, None]  # (batch, channel_64, 1, 1)
#         time_embed_128 = self.time_mlp_128(time)[:, :, None, None]
#         time_embed_256 = self.time_mlp_256(time)[:, :, None, None]

#         # 2. 编码过程
#         x1 = self.inc(x) + time_embed_64
#         x2 = self.down1(x1) + time_embed_128
#         x3 = self.down2(x2)  # 这里 x3 还未加 time_embed_256

#         # 3. Bottleneck
#         x3 = x3 + time_embed_256.repeat(1, 1, x3.shape[2], x3.shape[3])
#         bot1_out = self.bot1(x3)
#         bot2_out = self.bot2(bot1_out)
#         bot3_out = self.bot3(bot2_out)

#         # 4. 计算 up2_out（避免 UnboundLocalError）
#         _, _, H, W = bot3_out.shape  
#         time_embed_128 = time_embed_128.repeat(1, 1, H, W)  # 让时间嵌入匹配空间尺寸

#         up2_out = self.up2(torch.cat([bot3_out, x2], dim=1)) + time_embed_128

#         # 5. 计算 up3_out（同理）
#         _, _, H, W = up2_out.shape  
#         time_embed_64 = time_embed_64.repeat(1, 1, H, W)

#         up3_out = self.up3(torch.cat([up2_out, x1], dim=1)) + time_embed_64

#         # 6. 输出
#         output = self.outc(up3_out)
#         return output

# 2-blocks
class UNet(BaseNet):
    """
    UNet (Modified: Fewer Downsampling and Upsampling Layers)
    """

    def __init__(self, **kwargs):
        super(UNet, self).__init__(**kwargs)

        # Input Convolution
        self.inc = DoubleConv(in_channels=self.in_channel, out_channels=self.channel[1], act=self.act)

        # Downsampling layers
        self.down1 = DownBlock(in_channels=self.channel[1], out_channels=self.channel[2], act=self.act)
        self.sa1 = SelfAttention(channels=self.channel[2], size=self.image_size_list[1], act=self.act)
        self.down2 = DownBlock(in_channels=self.channel[2], out_channels=self.channel[3], act=self.act)
        self.sa2 = SelfAttention(channels=self.channel[3], size=self.image_size_list[2], act=self.act)

        # Bottleneck (updated)
        self.bot1 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)
        self.bot2 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)
        self.bot3 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[2], act=self.act)  # Adjusted to match up2

        # Upsampling layers (adjusted)
        self.up2 = UpBlock(in_channels=self.channel[3], out_channels=self.channel[1], act=self.act)  # Previously channel[3]
        self.sa5 = SelfAttention(channels=self.channel[1], size=self.image_size_list[1], act=self.act)
        self.up3 = UpBlock(in_channels=self.channel[2], out_channels=self.channel[1], act=self.act)
        self.sa6 = SelfAttention(channels=self.channel[1], size=self.image_size_list[0], act=self.act)

        # Output Convolution
        self.outc = nn.Conv2d(in_channels=self.channel[1], out_channels=self.out_channel, kernel_size=1)

    def forward(self, x, time, y=None):
        time = time.unsqueeze(-1).type(torch.float)
        time = self.pos_encoding(time, self.time_channel)

        if y is not None:
            time += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, time)
        x2_sa = x2  # self.sa1(x2)
        x3 = self.down2(x2_sa, time)
        x3_sa = x3  # self.sa2(x3)

        # Bottleneck
        bot1_out = self.bot1(x3_sa)
        bot2_out = self.bot2(bot1_out)
        bot3_out = self.bot3(bot2_out)

        # Updated upsampling flow
        up2_out = self.up2(bot3_out, x2_sa, time)
        up2_sa_out = up2_out  # self.sa5(up2_out)
        up3_out = self.up3(up2_sa_out, x1, time)
        up3_sa_out = up3_out  # self.sa6(up3_out)
        output = self.outc(up3_sa_out)

        return output


# class UNet(BaseNet):
#     """
#     UNet
#     """

#     def __init__(self, **kwargs):
#         """
#         Initialize the UNet network
#         :param in_channel: Input channel
#         :param out_channel: Output channel
#         :param channel: The list of channel
#         :param time_channel: Time channel
#         :param num_classes: Number of classes
#         :param image_size: Adaptive image size
#         :param device: Device type
#         :param act: Activation function
#         """
#         super(UNet, self).__init__(**kwargs)

#         # channel: 3 -> 64
#         # size: size
#         self.inc = DoubleConv(in_channels=self.in_channel, out_channels=self.channel[1], act=self.act)

#         # channel: 64 -> 128
#         # size: size / 2
#         self.down1 = DownBlock(in_channels=self.channel[1], out_channels=self.channel[2], act=self.act)
#         # channel: 128
#         # size: size / 2
#         self.sa1 = SelfAttention(channels=self.channel[2], size=self.image_size_list[1], act=self.act)
#         # channel: 128 -> 256
#         # size: size / 4
#         self.down2 = DownBlock(in_channels=self.channel[2], out_channels=self.channel[3], act=self.act)
#         # channel: 256
#         # size: size / 4
#         self.sa2 = SelfAttention(channels=self.channel[3], size=self.image_size_list[2], act=self.act)
#         # channel: 256 -> 256
#         # size: size / 8
#         self.down3 = DownBlock(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)
#         # channel: 256
#         # size: size / 8
#         self.sa3 = SelfAttention(channels=self.channel[3], size=self.image_size_list[3], act=self.act)

#         # channel: 256 -> 512
#         # size: size / 8
#         self.bot1 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[4], act=self.act)
#         # channel: 512 -> 512
#         # size: size / 8
#         self.bot2 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[4], act=self.act)
#         # channel: 512 -> 256
#         # size: size / 8
#         self.bot3 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[3], act=self.act)

#         # channel: 512 -> 128   in_channels: up1(512) = down3(256) + bot3(256)
#         # size: size / 4
#         self.up1 = UpBlock(in_channels=self.channel[4], out_channels=self.channel[2], act=self.act)
#         # channel: 128
#         # size: size / 4
#         self.sa4 = SelfAttention(channels=self.channel[2], size=self.image_size_list[2], act=self.act)
#         # channel: 256 -> 64   in_channels: up2(256) = sa4(128) + sa1(128)
#         # size: size / 2
#         self.up2 = UpBlock(in_channels=self.channel[3], out_channels=self.channel[1], act=self.act)
#         # channel: 128
#         # size: size / 2
#         self.sa5 = SelfAttention(channels=self.channel[1], size=self.image_size_list[1], act=self.act)
#         # channel: 128 -> 64   in_channels: up3(128) = sa5(64) + inc(64)
#         # size: size
#         self.up3 = UpBlock(in_channels=self.channel[2], out_channels=self.channel[1], act=self.act)
#         # channel: 128
#         # size: size
#         self.sa6 = SelfAttention(channels=self.channel[1], size=self.image_size_list[0], act=self.act)

#         # channel: 64 -> 3
#         # size: size
#         self.outc = nn.Conv2d(in_channels=self.channel[1], out_channels=self.out_channel, kernel_size=1)

#     def forward(self, x, time, y=None):
#         """
#         Forward
#         :param x: Input
#         :param time: Time
#         :param y: Input label
#         :return: output
#         """
#         time = time.unsqueeze(-1).type(torch.float)
#         time = self.pos_encoding(time, self.time_channel)

#         if y is not None:
#             time += self.label_emb(y)

#         x1 = self.inc(x)
#         x2 = self.down1(x1, time)
#         x2_sa = self.sa1(x2)
#         x3 = self.down2(x2_sa, time)
#         x3_sa = self.sa2(x3)
#         x4 = self.down3(x3_sa, time)
#         x4_sa = self.sa3(x4)

#         bot1_out = self.bot1(x4_sa)
#         bot2_out = self.bot2(bot1_out)
#         bot3_out = self.bot3(bot2_out)

#         up1_out = self.up1(bot3_out, x3_sa, time)
#         up1_sa_out = self.sa4(up1_out)
#         up2_out = self.up2(up1_sa_out, x2_sa, time)
#         up2_sa_out = self.sa5(up2_out)
#         up3_out = self.up3(up2_sa_out, x1, time)
#         up3_sa_out = self.sa6(up3_out)
#         output = self.outc(up3_sa_out)
#         return output


if __name__ == "__main__":
    # Unconditional
    net = UNet(device="cpu", image_size=128)
    # Conditional
    # net = UNet(num_classes=10, device="cpu", image_size=128)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(1, 3, 128, 128)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t).shape)
    # print(net(x, t, y).shape)
