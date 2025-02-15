#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 22:33
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import re
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
import time
import torch.nn.functional as F
import torch
import logging
import coloredlogs
sys.path.append(os.path.dirname(sys.path[0]))
from config.choices import sample_choices, network_choices, act_choices, image_format_choices, parse_image_size_type
from config.version import get_version_banner
from utils.check import check_image_size
from utils.initializer import device_initializer, network_initializer, sample_initializer, generate_initializer
from utils.utils import plot_images, save_images, save_one_image_in_images, check_and_create_dir
from utils.checkpoint import load_ckpt
sys.path.append("/cpfs04/user/hanyujin/causal-dm/guided-diffusion")
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

def extract_number_from_path(path):
    filename = os.path.basename(path)
    
    match = re.search(r'ckpt_(\d+)\.pt', filename)
    
    if match:
        return match.group(1)
    else:
        return None

def generate(args):
    """
    Generating
    :param args: Input parameters
    :return: None
    """
    logger.info(msg="Start generation.")
    logger.info(msg=f"Input params: {args}")
    # Weight path
    weight_path = args.weight_path
    # Run device initializer
    device = device_initializer(device_id=args.use_gpu)
    noise_steps = args.noise_steps
    sample_steps = args.sample_steps
    # Enable conditional generation, sample type, network, image size, number of classes and select activation function
    conditional, network, image_size, num_classes, act = generate_initializer(ckpt_path=weight_path, args=args,
                                                                              device=device)
    # Check image size format
    image_size = check_image_size(image_size=image_size)
    # Generation name
    generate_name = args.generate_name
    # Sample
    sample = args.sample
    # Number of images
    num_images = args.num_images
    # Use ema
    use_ema = args.use_ema
    # Format of images
    image_format = args.image_format
    # Saving path
    gen_epoch = extract_number_from_path(args.weight_path)
    result_path = os.path.join(args.result_path,f'epoch_{gen_epoch}_cfg{args.cfg_scale}_class{os.path.basename(args.classifier_path)}_{str(time.time())}_{args.time_schedule}')
    # Check and create result path
    check_and_create_dir(result_path)
    # Network
    Network = network_initializer(network=network, device=device)
    # Initialize the diffusion model
    if sample == 'ddpm':
        diffusion = sample_initializer(sample=sample, image_size=image_size, device=device, noise_steps = noise_steps, sample_steps = sample_steps)
    else:
        diffusion = sample_initializer(sample=sample, image_size=image_size, device=device, noise_steps = noise_steps, sample_steps = sample_steps)
    
    # diffusion = sample_initializer(sample=sample, image_size=image_size, device=device,noise_steps = noise_steps, sample_steps = sample_steps)
    # Is it necessary to expand the image?
    input_image_size = check_image_size(image_size=args.image_size)
    if image_size == input_image_size:
        new_image_size = None
    else:
        new_image_size = input_image_size
    
    # print("image_size",args.image_size)
    # logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    classifier = classifier.to(device)

    def cond_fn(x, t,classifier, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] #* args.cfg_scale

    # def model_fn(x, t, y=None):
    #     assert y is not None
    #     return model(x, t, y if args.class_cond else None)


    # Initialize model
    if conditional:
        # Generation class name
        class_name = args.class_name
        # classifier-free guidance interpolation weight
        cfg_scale = args.cfg_scale
        model = Network(num_classes=num_classes, device=device, image_size=image_size, act=act).to(device)
        load_ckpt(ckpt_path=weight_path, model=model, device=device, is_train=False, is_use_ema=use_ema,
                  conditional=conditional)
        if class_name == -1:
            y = torch.arange(num_classes).long().to(device)
            num_images = num_classes
        else:
            y = torch.Tensor([class_name] * num_images).long().to(device)
        x = diffusion.contrastive_sample(model=model, n=num_images, labels=y, classifier=classifier,cond_fn=cond_fn, conditional = True,cfg_scale = args.cfg_scale)
    else:
        class_name = args.class_name
        if class_name == -1:
            y = torch.arange(num_classes).long().to(device)
            num_images = num_classes
        else:
            y = torch.Tensor([class_name] * num_images).long().to(device)
        if network.startswith("DiT") or network.startswith("SiT"):
            model = Network(device=device, input_size=image_size).to(device)  
        else:
            model = Network(device=device, image_size=image_size, act=act).to(device)
        load_ckpt(ckpt_path=weight_path, model=model, device=device, is_train=False, conditional=conditional)
        x = diffusion.contrastive_sample(model=model, n=num_images, labels=y, classifier=classifier,cond_fn=cond_fn, conditional = False,time_schedule=args.time_schedule)

    if result_path == "" or result_path is None:
        plot_images(images=x)
    else:
        save_images(images=x, path=os.path.join(result_path, f"{generate_name}.{image_format}"))
        save_one_image_in_images(images=x, path=result_path, generate_name=generate_name, image_size=new_image_size,
                                 image_format=image_format)
        plot_images(images=x)
    logger.info(msg="Finish generation.")


if __name__ == "__main__":
    # Generating model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Generation name (required)
    parser.add_argument("--generate_name", type=str, default="df")
    # Input image size (required)
    # [Warn] Compatible with older versions
    # [Warn] Version <= 1.1.1 need to be equal to model's image size, version > 1.1.1 can set whatever you want
    parser.add_argument("--image_size", type=parse_image_size_type, default=64)
    # Generated image format
    # Recommend to use png for better generation quality.
    # Option: jpg/png
    parser.add_argument("--image_format", type=str, default="png", choices=image_format_choices)
    # Number of generation images (required)
    # if class name is `-1` and conditional `is` True, the model would output one image per class.
    parser.add_argument("--num_images", type=int, default=8)
    # Use ema model
    # If set to false, the pt file of the ordinary model will be used
    # If true, the pt file of the ema model will be used
    parser.add_argument("--use_ema", default=False, action="store_true")
    # Weight path (required)
    parser.add_argument("--weight_path", type=str, default="/your/path/Defect-Diffusion-Model/weight/model.pt")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results/vis")
    # Set the sample type (required)
    # If not set, the default is for 'ddpm'. You can set it to either 'ddpm', 'ddim' or 'plms'.
    # Option: ddpm/ddim/plms
    parser.add_argument("--sample", type=str, default="ddpm", choices=sample_choices)
    parser.add_argument("--noise_steps", type=int, default= 1000)
    parser.add_argument("--sample_steps", type=int, default= 100)
    # Batch size for training (required)
    # =====================Enable the conditional generation (if '--conditional' is set to 'True')=====================
    # Class name (required)
    # if class name is `-1`, the model would output one image per class.
    # [Note] The setting range should be [0, num_classes - 1].
    parser.add_argument("--class_name", type=int, default=0)
    # classifier-free guidance interpolation weight, users can better generate model effect (recommend)
    parser.add_argument("--cfg_scale", type=int, default=1)
    parser.add_argument("--use_gpu", type=int, default=0)

    # =====================Older versions(version <= 1.1.1)=====================
    # Enable conditional generation (required)
    # If enabled, you can modify the custom configuration.
    # For more details, please refer to the boundary line at the bottom.
    # [Note] The conditional settings are consistent with the loaded model training settings.
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's network, version > 1.1.1 can set whatever you want
    parser.add_argument("--conditional", default=False, action="store_true")
    # Set network
    # Option: unet/cspdarkunet
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's network, version > 1.1.1 can set whatever you want
    parser.add_argument("--network", type=str, default="unet", choices=network_choices)
    # Set activation function (needed)
    # [Note] The activation function settings are consistent with the loaded model training settings.
    # [Note] If you do not set the same activation function as the model, mosaic phenomenon will occur.
    # Option: gelu/silu/relu/relu6/lrelu
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's act, version > 1.1.1 can set whatever you want
    parser.add_argument("--act", type=str, default="gelu", choices=act_choices)
    # Number of classes (required)
    # [Note] The classes settings are consistent with the loaded model training settings.
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's num classes, version > 1.1.1 can set whatever you want
    parser.add_argument("--num_classes", type=int, default=3)


# Classifier-specific arguments
    parser.add_argument("--classifier_attention_resolutions", type=str, default="16",
                        help="The attention resolutions for the classifier model.")
    parser.add_argument("--classifier_depth", type=int, default=2,
                        help="The number of layers in the classifier.")
    parser.add_argument("--classifier_width", type=int, default=128,
                        help="The width of the classifier (number of channels in each layer).")
    parser.add_argument("--classifier_pool", type=str, default="attention", choices=["attention", "avg", "max"],
                        help="Pooling mechanism for the classifier (e.g., attention, average pooling, max pooling).")
    parser.add_argument("--classifier_type", type=str, default="unet", choices=["unet", "mlp"],
                        help="Pooling mechanism for the classifier (e.g., attention, average pooling, max pooling).")
    parser.add_argument("--classifier_resblock_updown", type=bool, default=True,
                        help="Whether to use residual blocks with up/down sampling in the classifier.")
    parser.add_argument("--classifier_use_scale_shift_norm", type=bool, default=True,
                        help="Whether to use scale-shift normalization in the classifier.")
    parser.add_argument("--classifier_use_fp16", type=bool, default=True,
                        help="Whether to use mixed precision (fp16) training for the classifier.")
    parser.add_argument("--classifier_path", type=str, default="/your/path/Defect-Diffusion-Model/weight/model.pt")
    parser.add_argument("--time_schedule", type=str, default="linear")
    args = parser.parse_args()
    # Get version banner
    get_version_banner()
    generate(args)
