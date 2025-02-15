#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import logging
import coloredlogs
import torchvision
from tqdm import tqdm
from utils.utils import plot_images, save_images, save_one_image_in_images, check_and_create_dir
from model.samples.base import BaseDiffusion
import os
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

class DDPMDiffusion(BaseDiffusion):
    """
    DDPM class
    """

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=2e-2, img_size=None, device="cpu",
                 schedule_name="linear"):
        """
        The implement of DDPM
        Paper: Denoising Diffusion Probabilistic Models
        URL: https://arxiv.org/abs/2006.11239
        :param noise_steps: Noise steps
        :param beta_start: β start
        :param beta_end: β end
        :param img_size: Image size
        :param device: Device type
        :param schedule_name: Prepare the noise schedule name
        """

        super().__init__(noise_steps, beta_start, beta_end, img_size, device, schedule_name)
    


    def sample(self, model, n,phase,labels=None, cfg_scale=None, save_intermediate=False, intermediate_steps=None, result_path=None):
        """
        DDPM sample method with optional intermediate results saving.
        :param model: Model
        :param n: Number of sample images
        :param labels: Labels
        :param cfg_scale: Classifier-free guidance interpolation weight
        :param save_intermediate: Whether to save intermediate results
        :param intermediate_steps: List of specific steps at which to save intermediate results
        :param result_path: Path to save the intermediate results as .npz files
        :return: Sample images
        """
        logger.info(msg=f"DDPM Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # Input dim: [n, 3, img_size_h, img_size_w]
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
            if save_intermediate and 1000 in intermediate_steps:
                x_flattened = x.view(n, -1).cpu().numpy()  # Move to CPU and flatten
                step_file_name = "prediction_1000.npz"
                step_file_path = os.path.join(result_path, step_file_name)
                np.savez(step_file_path, x=x_flattened)
                logger.info(f"Saved intermediate step at 1000 to {step_file_path}")
            
            # 'reversed(range(1, self.noise_steps))' iterates over a sequence of integers in reverse
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, total=self.noise_steps - 1):
                # Time step, creating a tensor of size n
                t = (torch.ones(n) * i).long().to(self.device)
                
                # Conditional input if available
                if labels is None and cfg_scale is None:
                    predicted_noise = model(x, t)
                else:
                    predicted_noise = model(x, t, labels)
                    # Classifier-free guidance
                    if cfg_scale > 0:
                        unconditional_predicted_noise = model(x, t, None)
                        predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cfg_scale)

                # Expand to a 4-dimensional tensor, and get the value according to the time step t
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Noise generation
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                # if i >= phase:
                #     noise = torch.zeros_like(x)  # 不再加入噪声
                # else:
                #     noise = torch.randn_like(x)  # 加入噪声

                # Update x according to the DDPM algorithm
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    
                # Save intermediate results if required
                if save_intermediate and i in intermediate_steps:
                    # Flatten x to shape (n, M)
                    x_flattened = x.view(n, -1).cpu().numpy()  # Move to CPU and flatten
                    step_file_name = f"prediction_{i}.npz"
                    step_file_path = os.path.join(result_path, step_file_name)
                    np.savez(step_file_path, x=x_flattened)
                    logger.info(f"Saved intermediate step at {i} to {step_file_path}")

        model.train()
        
        # Return the final result after processing
        x = (x.clamp(-1, 1) + 1) / 2  # Rescale to [0, 1]
        x = (x * 255).type(torch.uint8)  # Convert to 8-bit integer image
        return x
    
    def sample_hidden(self, model, n,phase, class_model,device,labels=None, cfg_scale=None):
        """
        DDPM sample method
        :param model: Model
        :param n: Number of sample images
        :param labels: Labels
        :param cfg_scale: classifier-free guidance interpolation weight
        :return: Sample images
        """
        model.eval()
        with torch.no_grad():
            # Initialize random noise
            # x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
            # mean_tensor = torch.load('/cpfs04/user/hanyujin/causal-dm/latent_class/best_k_14_means.pt')
            # mean_sample = mean_tensor[0].reshape(3, self.img_size[0], self.img_size[1])
            # cov_tensor = 0.1*torch.ones_like(mean_sample)  
            # mean_expanded = mean_sample.unsqueeze(0).expand(n, -1, -1, -1) 
            # cov_expanded = cov_tensor.unsqueeze(0).expand(n, -1, -1, -1)  
            # x = torch.normal(mean=mean_expanded, std=cov_expanded).to(self.device)
            score = np.load("/cpfs04/user/hanyujin/causal-dm/results/sunshadow_lfd_lnd_rfd_rnd/vis/epoch_400_1737189887.87018/prediction_1000.npz")['label']
            x = np.load("/cpfs04/user/hanyujin/causal-dm/results/sunshadow_lfd_lnd_rfd_rnd/vis/epoch_400_1737189887.87018/prediction_1000.npz")['x']
            x_ = np.load(f"/cpfs04/user/hanyujin/causal-dm/results/sunshadow_lfd_lnd_rfd_rnd/vis/epoch_400_1737189887.87018/prediction_{phase}.npz")['x']
            mask = (score >= 0.97) & (score <= 1.03)
            filtered_x = x[mask]
            filtered_x_ = x_[mask]
            target_size = n
            repeats = target_size // filtered_x.shape[0] + 1  
            x_repeated = np.repeat(filtered_x, repeats, axis=0)[:target_size] 
            x_repeated_ = np.repeat(filtered_x_, repeats, axis=0)[:target_size]

            x_repeated_tensor = torch.from_numpy(x_repeated)
            x_reshaped = x_repeated_tensor.view((target_size, 3, self.img_size[0], self.img_size[1]))  # 转换形状，确保输入的维度是正确的
            x_repeated_tensor_ = torch.from_numpy(x_repeated_)
            x_reshaped_ = x_repeated_tensor_.view((target_size, 3, self.img_size[0], self.img_size[1])).to(self.device)  

            # Step 3: 计算均值和设定固定的标准差
            mean_tensor = x_reshaped.mean(dim=0)  
            std_tensor = torch.ones_like(mean_tensor)   

            x = torch.normal(mean=mean_tensor.unsqueeze(0).expand(target_size, -1, -1, -1),
                                    std=std_tensor.unsqueeze(0).expand(target_size, -1, -1, -1)).to(self.device)  
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, total=self.noise_steps - 1):
                t = (torch.ones(n) * i).long().to(self.device)
                if labels is None and cfg_scale is None:
                    predicted_noise = model(x, t)
                else:
                    predicted_noise = model(x, t, labels)
                    if cfg_scale > 0:
                        unconditional_predicted_noise = model(x, t, None)
                        predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cfg_scale)

                # Get alpha, alpha_hat, beta at the current time step
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Noise for t > 1
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                # if i >= 500:
                #     noise = torch.zeros_like(x)  # 不再加入噪声
                # else:
                #     noise = torch.randn_like(x)  # 加入噪声

                # At i == 900, apply the given update formula
                if i == -1:
                    x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                    
                    # Flatten x before passing it to the neural network
                    scaler = StandardScaler()
                    original_shape = x.shape
                    x_tranfer = x.cpu().numpy() 
                    x_tranfer = x_tranfer.reshape(x_tranfer.shape[0], -1)
                    x_tranfer = scaler.fit_transform(x_tranfer)
                    x_tranfer = x_tranfer.reshape(original_shape)
                    x_flattened = torch.tensor(x_tranfer).float().view(original_shape[0], -1).to(device)  # Flatten to (n_samples, n_features)

                    # Pass the flattened data through the neural network
                    # simple_nn = SimpleNN(input_dim=x_flattened.size(1), output_dim=2)  # Assuming output_dim=2 for example
                    logits = class_model(x_flattened)
                    probabilities = F.softmax(logits, dim=1) 
                    # print("predicted_labels:",probabilities )
                    # predicted_labels = torch.argmax(probabilities , dim=1) 
                    # print("After predicted_labels:",predicted_labels)
                    # print("After predicted_labels:",len(predicted_labels))
                    # Only keep data with label == 1
                    first_column = probabilities[:, 0]
                    sorted_indices = torch.argsort(first_column, descending=True)
                    top_50_percent_indices = sorted_indices[:len(sorted_indices) // 10]
                    mask = torch.zeros(probabilities.shape[0], dtype=torch.bool, device=probabilities.device)  # 创建全 False 的 mask
                    mask[top_50_percent_indices] = True

                    # mask = predicted_labels == 0
                    # print("Before x.shape:",x.shape)
                    x = x[mask]
                    # print("After x.shape:",x.shape)
                    n = len(probabilities[mask])
                    # Optionally handle the outputs of the network as needed
                    # For example, use the outputs for further processing

                else:
                    # Standard DDPM update (only happens after i == 900)
                    x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                if i == phase:
                    x = x_reshaped_ 
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]
        x = (x * 255).type(torch.uint8)  # Convert to uint8 range
        return x

    # def sample(self, model, n, labels=None, cfg_scale=None):
    #     """
    #     DDPM sample method
    #     :param model: Model
    #     :param n: Number of sample images
    #     :param labels: Labels
    #     :param cfg_scale: classifier-free guidance interpolation weight, users can better generate model effect.
    #     Avoiding the posterior collapse problem, Reference paper: 'Classifier-Free Diffusion Guidance'
    #     :return: Sample images
    #     """
    #     logger.info(msg=f"DDPM Sampling {n} new images....")
    #     model.eval()
    #     with torch.no_grad():
    #         # Input dim: [n, 3, img_size_h, img_size_w]
    #         x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
    #         # 'reversed(range(1, self.noise_steps)' iterates over a sequence of integers in reverse
    #         for i in tqdm(reversed(range(1, self.noise_steps)), position=0, total=self.noise_steps - 1):
    #             # Time step, creating a tensor of size n
    #             t = (torch.ones(n) * i).long().to(self.device)
    #             # Whether the network has conditional input, such as multiple category input
    #             if labels is None and cfg_scale is None:
    #                 # Images and time steps input into the model
    #                 predicted_noise = model(x, t)
    #             else:
    #                 predicted_noise = model(x, t, labels)
    #                 # Avoiding the posterior collapse problem and better generate model effect
    #                 if cfg_scale > 0:
    #                     # Unconditional predictive noise
    #                     unconditional_predicted_noise = model(x, t, None)
    #                     # 'torch.lerp' performs linear interpolation between the start and end values
    #                     # according to the given weights
    #                     # Formula: input + weight * (end - input)
    #                     predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cfg_scale)
    #             # Expand to a 4-dimensional tensor, and get the value according to the time step t
    #             alpha = self.alpha[t][:, None, None, None]
    #             alpha_hat = self.alpha_hat[t][:, None, None, None]
    #             beta = self.beta[t][:, None, None, None]
    #             # Only noise with a step size greater than 1 is required.
    #             # For details, refer to line 3 of Algorithm 2 on page 4 of the paper
    #             if i > 1:
    #                 noise = torch.randn_like(x)
    #             else:
    #                 noise = torch.zeros_like(x)
    #             # In each epoch, use x to calculate t - 1 of x
    #             # For details, refer to line 4 of Algorithm 2 on page 4 of the paper
    #             x = 1 / torch.sqrt(alpha) * (
    #                     x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
    #                 beta) * noise
    #                             # Save intermediate results every 10 steps (if requested)

    #     model.train()
    #     # Return the value to the range of 0 and 1
    #     x = (x.clamp(-1, 1) + 1) / 2
    #     # Multiply by 255 to enter the effective pixel range
    #     x = (x * 255).type(torch.uint8)
    #     return x
    
    def sample_step(self, model, n, output_dir, image_size,save_intermediate = None, labels=None, cfg_scale=None):
        """
        DDPM sample method
        :param model: Model
        :param n: Number of sample images
        :param labels: Labels
        :param cfg_scale: classifier-free guidance interpolation weight, users can better generate model effect.
        Avoiding the posterior collapse problem, Reference paper: 'Classifier-Free Diffusion Guidance'
        :return: Sample images
        """
        logger.info(msg=f"DDPM Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # Input dim: [n, 3, img_size_h, img_size_w]
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
            # 'reversed(range(1, self.noise_steps)' iterates over a sequence of integers in reverse
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, total=self.noise_steps - 1):
                # Time step, creating a tensor of size n
                t = (torch.ones(n) * i).long().to(self.device)
                # Whether the network has conditional input, such as multiple category input
                if labels is None and cfg_scale is None:
                    # Images and time steps input into the model
                    predicted_noise = model(x, t)
                else:
                    predicted_noise = model(x, t, labels)
                    # Avoiding the posterior collapse problem and better generate model effect
                    if cfg_scale > 0:
                        # Unconditional predictive noise
                        unconditional_predicted_noise = model(x, t, None)
                        # 'torch.lerp' performs linear interpolation between the start and end values
                        # according to the given weights
                        # Formula: input + weight * (end - input)
                        predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cfg_scale)
                # Expand to a 4-dimensional tensor, and get the value according to the time step t
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                # Only noise with a step size greater than 1 is required.
                # For details, refer to line 3 of Algorithm 2 on page 4 of the paper
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # In each epoch, use x to calculate t - 1 of x
                # For details, refer to line 4 of Algorithm 2 on page 4 of the paper
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
                                
                if save_intermediate and i % save_intermediate == 0:
                    # Get the current generated image as x_0 (after reversing the noise process)
                    x_0 = (x.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]
                    x_0 = (x_0* 255).type(torch.uint8)# Convert to [0, 255] range for saving
                    save_one_image_in_images(images=x_0, path=output_dir, generate_name=f'step_{self.noise_steps - i}', image_size=image_size,
                                 image_format='png')
                    
        
        model.train()
        # Return the value to the range of 0 and 1
        x = (x.clamp(-1, 1) + 1) / 2
        # Multiply by 255 to enter the effective pixel range
        x = (x * 255).type(torch.uint8)
        return x
    
    def contrastive_sample(self, model, n, labels=None,classifier=None, cond_fn=None, conditional = False,time_schedule='linear',cfg_scale=7):
        """
        DDPM sample method with optional classifier guidance.

        :param model: Diffusion model
        :param n: Number of sample images
        :param labels: Target labels for guidance
        :param cfg_scale: Classifier-free guidance interpolation weight
        :param cond_fn: Optional classifier guidance function
        :return: Sample images
        """
        logger.info(msg=f"DDPM Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # Initialize the latent variable x with Gaussian noise
            # x = torch.randn((n, self.image_channel, self.img_size[0], self.img_size[1])).to(self.device)
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, total=self.noise_steps - 1):
                t = (torch.ones(n) * i).long().to(self.device)

                # Compute the predicted mean and variance (from the diffusion model)
                model_out = model(x, t, labels if conditional else None)
                if isinstance(model_out, tuple):  # Support models with multiple outputs
                    predicted_noise, predicted_variance = model_out
                else:
                    predicted_noise = model_out
                    predicted_variance = self.beta[t][:, None, None, None]

                # Compute posterior mean and variance
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_prev = self.alpha_hat[t-1][:, None, None, None] if i > 1 else torch.ones_like(alpha_hat)
                posterior_variance = beta * (1. - alpha_hat_prev) / (1. - alpha_hat)
                posterior_mean = 1 / torch.sqrt(alpha) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
                )

                # Apply classifier guidance (correct the posterior mean)
                # print(" A cond_fn:",cond_fn,"labels:",labels )
                progress = 1.0 - (i / self.noise_steps) 
                if time_schedule == 'linear':
                    w_t = cfg_scale + 3.0 * progress  # 从7线性增长到10（当progress=1时）
                elif time_schedule == 'quadratic':
                    w_t = cfg_scale + 3.0 * (progress ** 2)  # 二次增长更平缓
                elif time_schedule == 'exponential':
                    w_t = cfg_scale + 3.0 * (1 - torch.exp(-5 * progress))  # 指数增长
                elif time_schedule == 'pisewise':
                    if i <= 20:  # 最后20步开始增强
                        w_t = cfg_scale  * ((20 - i)/20)  # 最后20步从7线性增长到10
                    else:
                        w_t = 0 #cfg_scale
                else:
                    w_t = cfg_scale  # 默认固定值
                if cond_fn is not None and labels is not None:
                    # print(" B cond_fn:",cond_fn,"labels:",labels )
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        gradient = cond_fn(x = x_in, t=t, y=labels,classifier=classifier)


                    # print("posterior_mean:",posterior_mean)
                    # print("posterior_variance:",posterior_variance,posterior_variance.shape)
                    # print("gradient:", gradient.shape)
                    # print("gradient:", posterior_variance * gradient)
                    posterior_mean = posterior_mean + w_t * posterior_variance * gradient
                    

                # Sample from the posterior
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = posterior_mean + torch.sqrt(posterior_variance) * noise

            model.train()

        # Post-process the generated images
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


