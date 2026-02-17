import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
from collections import deque

import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode,
    setup_seed, init_model, SkipBatchSampler
)

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    # https://github.com/jingyaogong/minimind/issues/298
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # 防止零长度mask导致除零NaN
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def grad_norm(model, norm_type=2.0):
    # 只计算，不修改梯度
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    device = parameters[0].grad.device
    norms = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
    total = torch.norm(norms, norm_type)
    return total.item()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = 0.0

    # small moving averages (min-change logging improvement)
    ma_win = 20
    ma_loss = deque(maxlen=ma_win)
    ma_dpo = deque(maxlen=ma_win)
    ma_aux = deque(maxlen=ma_win)
    ma_logits = deque(maxlen=ma_win)
    ma_acc = deque(maxlen=ma_win)
    ma_kl = deque(maxlen=ma_win)

    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # ===== DEBUG LOG (print once) =====
        if step == start_step + 1 and is_main_process():
            with torch.no_grad():
                ms = mask.sum(dim=1)
                Logger(
                    f"[debug] mask: dtype={mask.dtype}, shape={tuple(mask.shape)}, "
                    f"sum(min/mean/max)={ms.min().item():.1f}/{ms.float().mean().item():.1f}/{ms.max().item():.1f}, "
                    f"unique={torch.unique(mask).detach().cpu().tolist()[:10]}"
                )
                diff_ratio = (x_chosen != x_rejected).float().mean().item()
                Logger(f"[debug] x_chosen!=x_rejected token diff ratio: {diff_ratio:.4f}")
                y_unique = torch.unique(y).detach().cpu()
                Logger(f"[debug] y unique (first 20): {y_unique.tolist()[:20]}")
                ignore_count = (y == -100).sum().item()
                Logger(f"[debug] y==-100 count: {ignore_count} / {y.numel()}")
                zero_mask_rows = (ms == 0).sum().item()
                Logger(f"[debug] mask rows with sum==0: {zero_mask_rows} / {mask.size(0)}")

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            if step == start_step + 1 and is_main_process():
                with torch.no_grad():
                    ms = mask.sum(dim=1).clamp_min(1e-8)
                    ref_score = (ref_log_probs * mask).sum(dim=1) / ms
                    Logger(f"[debug] ref_score (first 8): {ref_score.detach().cpu().tolist()[:8]}")

            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        # gradient step
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            pre = grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            post = grad_norm(model)

            # log grad_norm only at log intervals (reduce spam)
            if is_main_process() and (step % args.log_interval == 0 or step == iters - 1):
                Logger(f"[stat] grad_norm pre={pre:.4f} post={post:.4f}")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # main logging
        if step % args.log_interval == 0 or step == iters - 1:
            now = time.time()
            elapsed = now - start_time
            step_done = max(step, 1)
            steps_left = max(iters - step, 0)
            eta_min = (elapsed / step_done) * steps_left / 60.0
            elapsed_min = elapsed / 60.0

            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']

            # sentence-level scores & stats
            with torch.no_grad():
                ms = mask.sum(dim=1).clamp_min(1e-8)
                ref_score = (ref_log_probs * mask).sum(dim=1) / ms
                pol_score = (policy_log_probs * mask).sum(dim=1) / ms

                bs = ref_score.size(0)
                cr, rr = ref_score[:bs // 2], ref_score[bs // 2:]
                cp, rp = pol_score[:bs // 2], pol_score[bs // 2:]

                pi_lr = (cp - rp)
                ref_lr = (cr - rr)
                dpo_logits = (pi_lr - ref_lr)
                pref_acc = (dpo_logits > 0).float().mean().item()

                # "KL-like" proxy (policy vs ref drift on both chosen & rejected)
                delta_c = (cp - cr)
                delta_r = (rp - rr)
                kl_proxy = 0.5 * (delta_c + delta_r)  # can be +/- ; square for magnitude
                kl_proxy_mean = kl_proxy.mean().item()
                kl_proxy_std = kl_proxy.std(unbiased=False).item()
                kl_proxy_l2 = (kl_proxy ** 2).mean().item()

                # throughput (tokens/sec) over last log interval
                tok_now = mask.sum().item()
                interval_tokens = tok_now + last_log_tokens  # approx since we don't accumulate separately
                interval_time = max(now - last_log_time, 1e-6)
                tps = tok_now / interval_time  # tokens/sec for this step batch (approx)
                last_log_time = now
                last_log_tokens = 0.0

            # moving averages
            ma_loss.append(current_loss)
            ma_dpo.append(current_dpo_loss)
            ma_aux.append(current_aux_loss)
            ma_logits.append(dpo_logits.mean().item())
            ma_acc.append(pref_acc)
            ma_kl.append(kl_proxy_l2)

            def _mean(dq):
                return float(sum(dq) / max(len(dq), 1))

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, "
                f"learning_rate: {current_lr:.8f}, elapsed: {elapsed_min:.3f}min, eta: {eta_min:.3f}min, "
                f"toks/s: {tps:.1f}, "
                f"ma({ma_win}) loss: {_mean(ma_loss):.4f}, logits: {_mean(ma_logits):.4f}, acc: {_mean(ma_acc):.3f}"
            )

            Logger(
                f"[stat] pi_lr mean={pi_lr.mean().item():.4f} std={pi_lr.std(unbiased=False).item():.4f} | "
                f"ref_lr mean={ref_lr.mean().item():.4f} std={ref_lr.std(unbiased=False).item():.4f} | "
                f"logits mean={dpo_logits.mean().item():.4f} std={dpo_logits.std(unbiased=False).item():.4f} | "
                f"pref_acc={pref_acc:.3f} | "
                f"kl_proxy mean={kl_proxy_mean:.4f} std={kl_proxy_std:.4f} l2={kl_proxy_l2:.6f}"
            )

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "dpo_loss": current_dpo_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "elapsed_min": elapsed_min,
                    "eta_min": eta_min,
                    "toks_per_s": tps,
                    "pi_lr_mean": pi_lr.mean().item(),
                    "pi_lr_std": pi_lr.std(unbiased=False).item(),
                    "ref_lr_mean": ref_lr.mean().item(),
                    "ref_lr_std": ref_lr.std(unbiased=False).item(),
                    "logits_mean": dpo_logits.mean().item(),
                    "logits_std": dpo_logits.std(unbiased=False).item(),
                    "pref_acc": pref_acc,
                    "kl_proxy_mean": kl_proxy_mean,
                    "kl_proxy_std": kl_proxy_std,
                    "kl_proxy_l2": kl_proxy_l2,
                    "ma_loss": _mean(ma_loss),
                    "ma_logits": _mean(ma_logits),
                    "ma_pref_acc": _mean(ma_acc),
                    "ma_kl_l2": _mean(ma_kl),
                })

        # checkpoint saving
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints'
            )
            model.train()
            del state_dict

        # cleanup
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int,
                        help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.1, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # 初始化参考模型（ref_model冻结）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()