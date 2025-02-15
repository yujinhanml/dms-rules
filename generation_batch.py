import os
import subprocess
import argparse
from glob import glob

def run_generation(dataset_name, gpu, model=None):
    # 基础路径
    base_path = f"/cpfs01/user/hanyujin/causal-dm/results/{dataset_name}"
    
    # 如果没有指定特定的模型，则搜索所有的 .pt 文件
    if model is None:
        pt_files = glob(os.path.join(base_path, "*.pt"))
    else:
        pt_files = [os.path.join(base_path, model)]
    generate_script_dir = os.path.dirname("/cpfs01/user/hanyujin/causal-dm/Integrated-Design-Diffusion-Model/tools/generate.py")
    # 对每个 .pt 文件运行生成命令
    for pt_file in pt_files:
        model_name = os.path.basename(pt_file).split('.')[0]  # 获取模型名称（不含.pt）
        epoch = model_name.split('_')[-1]  # 提取 epoch 数
        
        # 创建保存生成图片的目录
        save_dir = os.path.join(base_path, "vis", f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 构建并运行命令
        cmd = [
            "python", "generate.py",
            "--generate_name", "sunshadow",
            "--num_images", "3000",
            "--image_size", "32",
            "--weight_path", pt_file,
            "--sample", "ddpm",
            "--result_path", save_dir,
            "--use_gpu", str(gpu)
        ]
        
        print(f"Running generation for {model_name}...")
        subprocess.run(cmd, check=True, cwd=generate_script_dir)
        print(f"Generation complete for {model_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image generation for multiple checkpoints.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--gpu", type=int, default=1, help="GPU to use")
    parser.add_argument("--model", type=str, default=None, help="Specific model to use (optional)")
    
    args = parser.parse_args()
    
    run_generation(args.dataset_name, args.gpu, args.model)