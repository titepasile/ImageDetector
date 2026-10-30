"""Stage A, loop 2: train the local encoder on 128x128 native-resolution crops.

Supervised training : LocalResNet + temporary fc head, cross-entropy
on real/AI labels. After training, only the encoder part (model.features) is
used further; export_local_features.py writes the z2 vectors for stage B.

What happens per epoch:
  1. train   forward -> loss -> backward -> update, over all training crops
  2. val     predictions on the validation split, crop- and image-level
  3. save    last.pt every epoch, best.pt when the validation image AUC
             is the best so far (that is the model you hand over)
  4. log     console, OUT/metrics.jsonl and (optional) MLflow

--> validation, checkpoints, resuming, a per-step learning-rate schedule (warmup + cosine), mixed
precision on GPU, and per-image loss weights.

Usage:
    python train_local.py SHARDS_DIR runs/local_v1 --epochs 30 --batch-size 128
    python train_local.py SHARDS_DIR runs/local_v1 --resume      # continue
Smoke test on a few batches:
    python train_local.py SHARDS_DIR runs/smoke --epochs 3 --max-steps 20
"""

import argparse
import json
import random
import time
import mlflow
from pathlib import Path

import numpy as np
import torch

from src.data.local_dataset import make_loader
from src.model.local_res_net import LocalResNet
from src.local_training import (
    evaluate,
    pick_device,
    save_checkpoint,
    train_epoch,
    warmup_cosine,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--blocks", type=int, nargs="+", default=[2, 2, 2, 2])
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--first-stage-stride", type=int, default=1,
                        help="2 halves the cost of the first stage")
    parser.add_argument("--jpeg-aug-prob", type=float, default=0.0,
                        help="probability of random extra JPEG compression (both classes)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=6,
                        help="stop after this many epochs without val improvement (0 = never)")
    parser.add_argument("--max-steps", type=int, default=0, help="limit steps per epoch (smoke tests)")
    parser.add_argument("--resume", action="store_true", help="continue from OUT_DIR/last.pt")
    parser.add_argument("--mlflow", action="store_true", help="also log to MLflow")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    train_loader, train_set, n_train = make_loader(
        args.shards_dir, "train", batch_size=args.batch_size, num_workers=args.workers,
        jpeg_aug_prob=args.jpeg_aug_prob, seed=args.seed)
    val_loader, _, n_val = make_loader(
        args.shards_dir, "val", batch_size=args.batch_size * 2, num_workers=args.workers)
    if n_train is None:
        raise ValueError("manifest.json missing: needed to count training crops")
    steps_per_epoch = n_train // args.batch_size
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)

    model = LocalResNet(args.blocks, base_channels=args.base_channels,
                        first_stage_stride=args.first_stage_stride).to(device)
    # no weight decay on BatchNorm and biases (common practice)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, int(args.warmup_epochs * steps_per_epoch)))

    start_epoch, best_auc, epochs_without_improvement = 0, -1.0, 0
    best_score = (-1.0, -float("inf"))
    if args.resume and (out_dir / "last.pt").exists():
        checkpoint = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = tuple(checkpoint["metrics"].get("best_val_score", best_score))
        best_auc = best_score[0]
        epochs_without_improvement = checkpoint["metrics"].get("epochs_without_improvement", 0)
        print(f"Resumed from epoch {checkpoint['epoch']}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} amp={amp_dtype} params={n_params / 1e6:.2f}M "
          f"z2_dim={model.feature_dim} train_crops={n_train} val_crops={n_val} "
          f"steps/epoch={steps_per_epoch}")

    if args.mlflow:
        mlflow.start_run(run_name=out_dir.name)
        mlflow.log_params({k: str(v) for k, v in vars(args).items()})

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        train_set.set_epoch(epoch)
        metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler,
                              device, amp_dtype, args.max_steps)
        val_metrics, _ = evaluate(model, val_loader, device, amp_dtype, max_steps=args.max_steps)
        metrics.update(val_metrics)
        metrics["epoch"] = epoch
        metrics["seconds"] = round(time.time() - started, 1)

        # Image-level AUC decides which model is "best" (accuracy as fallback).
        # AUC often saturates near 1.0, so on a tie the lower val loss wins.
        score = (round(metrics.get("val_image_auc", metrics.get("val_image_accuracy", 0.0)), 4),
                 -metrics.get("val_loss", 0.0))
        improved = score > best_score
        if improved:
            best_score, epochs_without_improvement = score, 0
        else:
            epochs_without_improvement += 1
        best_auc = best_score[0]
        metrics["best_val_score"] = list(best_score)
        metrics["best_val_image_auc"] = best_auc
        metrics["epochs_without_improvement"] = epochs_without_improvement

        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, metrics, args)
        if improved:
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, metrics, args)
        with (out_dir / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(metrics) + "\n")
        if mlflow:
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float)) and not isinstance(v, bool)}, step=epoch)
        print(f"epoch {epoch:3d}  loss {metrics['train_loss']:.4f}  "
              f"train_acc {metrics['train_crop_accuracy']:.3f}  "
              f"val_img_acc {metrics.get('val_image_accuracy', float('nan')):.3f}  "
              f"val_img_auc {metrics.get('val_image_auc', float('nan')):.3f}  "
              f"{'*best*' if improved else ''}  ({metrics['seconds']}s)")

        if args.patience and epochs_without_improvement >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping early.")
            break

    if mlflow:
        mlflow.end_run()
    print(f"Best validation image AUC: {best_auc:.4f}  ->  {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
