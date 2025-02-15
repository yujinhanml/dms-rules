#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/4/22 18:07
    @Author : egoist945402376
    @Site   : https://github.com/chairc
"""
import subprocess


def run_fid_command(generated_image_folder, dataset_image_folder, dim=2048):

    command = f'python -m pytorch_fid {generated_image_folder} {dataset_image_folder} --dim={dim}'

    try:

        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

        print("Successfully calculate FID score!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to calculate the FID score:")
        print(e.stderr)


# Modify your path to dataset images folder and generated images folder.
generated_image_folder = "/cpfs04/user/hanyujin/causal-dm/Integrated-Design-Diffusion-Model/datasets/sunshadow_lfd_lnd_rfd_rnd_32/images"
dataset_image_folder = "/cpfs04/user/hanyujin/causal-dm/results/sunshadow_lfd_lnd_rfd_rnd/vis/epoch_400_1733622674.3212345"


# dimensions options: 768/ 2048
# Choose 768 for fast calculation and reduced memory requirement
# choose 2048 for better calculation
# default: 2048
dimensions = 2048
# run
run_fid_command(generated_image_folder, dataset_image_folder, dimensions)
