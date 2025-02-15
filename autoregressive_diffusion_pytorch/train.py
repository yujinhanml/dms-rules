import torch
import os
import sys
import argparse
from accelerate import Accelerator
sys.path.append('/cpfs04/user/hanyujin/causal-dm/AR_diff/autoregressive-diffusion-pytorch')
from autoregressive_diffusion_pytorch import (
    ImageDataset,
    ImageAutoregressiveDiffusion,
    ImageTrainer
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Autoregressive Diffusion model")
    parser.add_argument("--data_path", type=str, default='/cpfs04/user/hanyujin/causal-dm/Integrated-Design-Diffusion-Model/datasets/sunshadow_lfd_lnd_rfd_rnd', help="Path to the dataset")
    parser.add_argument("--kl_path", type=str, default=None, help="Path to KL-VAE")
    parser.add_argument("--image_size", type=int, default=32, help="Size of the images")
    parser.add_argument("--patch_size", type=int, default=16, help="Size of the patches")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels in the images")
    parser.add_argument("--dim", type=int, default=480, help="Dimension of the model")
    parser.add_argument("--depth", type=int, default=6, help="Dimension of the model")
    parser.add_argument("--mlp_depth", type=int, default=3, help="Dimension of the model")
    parser.add_argument("--num_train_steps", type=int, default=125*2*500, help="Number of training steps")
    parser.add_argument("--use_gpu", type=int, default=0, help="GPU")
    parser.add_argument("--save_results_every", type=int, default=5000, help="Save results every N steps")
    parser.add_argument("--checkpoint_every", type=int, default=125*2*100, help="Save checkpoint every N steps")
    parser.add_argument('--use_parallel_order', action='store_true', help='Use spiral order for token generation')
    parser.add_argument('--no_weight', action='store_true', help='Use spiral order for token generation')
    parser.add_argument('--reverse', action='store_true', help='Use reverse order for token generation')
    return parser.parse_args()

def main():
    args = parse_args()
    device = f'cuda:{args.use_gpu}' if args.use_gpu >= 0 else 'cpu'
    # Dataset setup
    dataset = ImageDataset(
        os.path.join(args.data_path, 'images'),
        image_size=args.image_size,
        reverse=args.reverse
    )
    dataset_name = os.path.basename(args.data_path)
    print(f"Dataset name: {dataset_name}")

    # Model setup
    model = ImageAutoregressiveDiffusion(
        image_size=args.image_size,
        patch_size=args.patch_size,
        channels=args.channels,
        model=dict(dim=args.dim,depth=args.depth,mlp_depth=args.mlp_depth),
        kl_path=args.kl_path,
        use_parallel_order = args.use_parallel_order,
        device = device
    )

    # Training
    trainer = ImageTrainer(
        model=model,
        dataset=dataset,
        num_train_steps=args.num_train_steps,
        checkpoints_folder=f'/cpfs04/user/hanyujin/causal-dm/AR_diff/results/{dataset_name}_parallel{args.use_parallel_order}_reverse{args.reverse}_patch{args.patch_size}_dim{args.dim}_depth{args.depth}_mlpdepth{args.mlp_depth}_noweightedloss{args.no_weight}_trainsteps{args.num_train_steps}/checkpoints',
        results_folder=f'/cpfs04/user/hanyujin/causal-dm/AR_diff/results/{dataset_name}_parallel{args.use_parallel_order}_reverse{args.reverse}_patch{args.patch_size}_dim{args.dim}_depth{args.depth}_mlpdepth{args.mlp_depth}_noweightedloss{args.no_weight}_trainsteps{args.num_train_steps}/checkimg',
        save_results_every=args.save_results_every,
        checkpoint_every=args.checkpoint_every,
        use_parallel_order = args.use_parallel_order,
        reverse=args.reverse,
        dataset_name = dataset_name,
        patch_size = args.patch_size,
        image_size = args.image_size,
        dim =args.dim,
        no_weight = args.no_weight,
        depth = args.depth,
        mlp_depth = args.mlp_depth,
        device = device
        # project_name="image_diffusion_parallel" if args.use_parallel_order else "image_diffusion_serial"
    )
    trainer()

if __name__ == "__main__":
    main()