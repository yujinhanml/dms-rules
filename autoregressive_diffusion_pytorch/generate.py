import torch
import os
import sys
import argparse
from torchvision.utils import save_image

sys.path.append('/cpfs04/user/hanyujin/causal-dm/AR_diff/autoregressive-diffusion-pytorch')
from autoregressive_diffusion_pytorch import ImageAutoregressiveDiffusion

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using trained Autoregressive Diffusion model")
    parser.add_argument("--dataset", type=str, default='sunshadow_lfd_lnd_rfd_rnd_parallelFalse_reverseFalse_patch16_dim480_depth3_mlpdepth3_noweightedlossTrue_trainsteps125000', help="Path to the dataset (for naming)")
    parser.add_argument("--image_size", type=int, default=32, help="Size of the images")
    parser.add_argument("--patch_size", type=int, default=16, help="Size of the patches")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels in the images")
    parser.add_argument("--depth", type=int, default=3, help="Dimension of the model")
    parser.add_argument("--mlp_depth", type=int, default=3, help="Dimension of the model")
    parser.add_argument("--dim", type=int, default=480, help="Dimension of the model")
    parser.add_argument("--num_generate", type=int, default=3000, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument('--use_parallel_order', action='store_true', help='Use spiral order for token generation')
    parser.add_argument("--model_pt", type=str, default='checkpoint.125000', help="Model checkpoint to load for generation")
    parser.add_argument(
        "--use_gpu", type=int, default=0, help="Specify the GPU number to use (e.g., 0, 1). If not specified, CPU is used."
    )
    return parser.parse_args()



def main():
    args = parse_args()
    # args.data_path = '/cpfs04/user/hanyujin/causal-dm/Integrated-Design-Diffusion-Model/datasets/' + args.dataset
    # dataset_name = os.path.basename(args.data_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{args.use_gpu}")
    # Model setup
    model = ImageAutoregressiveDiffusion(
        image_size=args.image_size,
        patch_size=args.patch_size,
        channels=args.channels,
        model=dict(dim=args.dim),
        use_parallel_order = args.use_parallel_order,
        device=device,
        depth = args.depth,
        mlp_depth = args.mlp_depth
    )

    # Load trained model
    checkpoint_path = f"/cpfs04/user/hanyujin/causal-dm/AR_diff/results/{args.dataset}/checkpoints/{args.model_pt}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    # Generate images
    save_dir = f"/cpfs04/user/hanyujin/causal-dm/AR_diff/results/{args.dataset}/generations/{args.model_pt}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generated {args.num_generate} images, saved in {save_dir}")
    with torch.no_grad():
        for start_idx in range(0, args.num_generate, args.batch_size):
            current_batch_size = min(args.batch_size, args.num_generate - start_idx)
            generated_images = model.sample(batch_size=current_batch_size)
            
            for j, image in enumerate(generated_images):
                image_index = start_idx + j
                save_path = os.path.join(save_dir, f"{image_index}.png")
                save_image(image, save_path, normalize=True)
            
            print(f"Generated and saved images {start_idx} to {start_idx + current_batch_size - 1}")

    # print(f"Generated {args.num_generate} images, saved in {save_dir}")

if __name__ == "__main__":
    main()