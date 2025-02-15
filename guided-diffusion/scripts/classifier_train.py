"""
Train a noised image classifier on ImageNet with wandb integration and optional Contrastive Loss.
"""

import argparse
import os, sys
import wandb  # 导入 wandb
sys.path.append("/cpfs04/user/hanyujin/causal-dm/guided-diffusion")
import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict


def nt_xent_loss(features, labels, temperature=0.5):
    """
    NT-Xent Loss for contrastive learning.

    Args:
        features: Embedding vectors (batch_size, feature_dim).
        labels: Corresponding labels (batch_size,).
        temperature: Temperature scaling factor.

    Returns:
        loss: Contrastive loss value.
    """
    features = F.normalize(features, dim=1)  # L2 normalize features
    similarity_matrix = th.matmul(features, features.T) / temperature  # Cosine similarity

    # Create labels mask
    labels = labels.unsqueeze(1)  # (batch_size, 1)
    mask = th.eq(labels, labels.T).float()  # Same class -> 1, Different class -> 0

    # Compute loss
    exp_similarity = th.exp(similarity_matrix)
    positive_sim = exp_similarity * mask  # Positive pairs
    negative_sim = exp_similarity * (1 - mask)  # Negative pairs

    positive_sum = positive_sim.sum(dim=1)
    negative_sum = negative_sim.sum(dim=1)

    loss = -th.log(positive_sum / (positive_sum + negative_sum + 1e-8))
    return loss.mean()

def main():
    args = create_argparser().parse_args()
    # 设置指定的 GPU
    device = torch.device(f"cuda:{args.use_gpu}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.use_gpu < 0 or args.use_gpu >= num_gpus:
            raise ValueError(
                f"Invalid GPU ID {args.use_gpu}. Available GPUs: {list(range(num_gpus))}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)
        device = torch.device("cuda")
    else:
        raise RuntimeError("No GPUs available. Please check your setup.")

    dist_util.setup_dist()
    logger.configure(args)
    last_part = os.path.basename(args.data_dir)

    # 初始化 wandb 项目
    if dist.get_rank() == 0:  # 只在主进程初始化 wandb
        wandb.init(
            project="guided-diffusion",  # 项目名称
            name=f"classifier_train_{args.image_size}_{last_part}_batch{args.batch_size}_iter{args.iterations}_contrastive_{args.contrastive}{args.contrastive_weight}_classifier_{args.classifier_type}",
            config=args_to_dict(args, classifier_and_diffusion_defaults().keys()),
        )

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(device)  # 将模型移到指定设备
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=device
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[device],
        output_device=device,
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=True,
    )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=device)
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(device)

        batch = batch.to(device)
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], device)
            batch = diffusion.q_sample(batch, t)
        else:
            t = torch.zeros(batch.shape[0], dtype=torch.long, device=device)

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t.float())

            # 分类损失
            loss_classification = F.cross_entropy(logits, sub_labels)

            # 对比损失（如果启用对比学习）
            if args.contrastive:
                features = model.module.extract_features(sub_batch, timesteps=sub_t.float())  # 提取特征
                loss_contrastive = nt_xent_loss(features, sub_labels, temperature=args.temperature)
                loss = loss_classification + args.contrastive_weight * loss_contrastive
            else:
                loss = loss_classification

            # 日志记录
            if dist.get_rank() == 0:
                wandb.log({
                    f"{prefix}_loss": loss.item(),
                    f"{prefix}_classification_loss": loss_classification.item(),
                    f"{prefix}_contrastive_loss": loss_contrastive.item() if args.contrastive else 0,
                    f"{prefix}_acc@1": compute_top_k(logits, sub_labels, k=1),
                    f"{prefix}_acc@2": compute_top_k(logits, sub_labels, k=2),
                    "step": step + resume_step
                })

            # 反向传播
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with torch.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()





def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)



def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        contrastive=False,  # 是否使用对比学习
        contrastive_weight=0.5,  # 对比损失权重
        temperature=0.5,  # 对比学习温度
        use_gpu=0,  # 新增参数：指定 GPU
        classifier_type= 'mlp'
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
