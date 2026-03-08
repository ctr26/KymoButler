"""Training entrypoint for KymoButler UNet segmentation models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import KymographDataset, SyntheticKymographGenerator
from .losses import CombinedLoss
from .models import build_unet, build_unet_unidirectional


def _foreground_prob(pred: Tensor) -> Tensor:
    if pred.shape[1] == 1:
        return pred
    return pred[:, 1:2, :, :]


def _dice_score(pred: Tensor, target: Tensor, eps: float = 1e-6) -> float:
    p = (_foreground_prob(pred) > 0.5).float()
    t = target.float()
    intersection = (p * t).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    score = (2.0 * intersection + eps) / (denom + eps)
    return float(score.mean().item())


def _iou_score(pred: Tensor, target: Tensor, eps: float = 1e-6) -> float:
    p = (_foreground_prob(pred) > 0.5).float()
    t = target.float()
    intersection = (p * t).sum(dim=(1, 2, 3))
    union = ((p + t) > 0).float().sum(dim=(1, 2, 3))
    score = (intersection + eps) / (union + eps)
    return float(score.mean().item())


def _compute_loss_and_metrics(
    outputs: Tensor | dict[str, Tensor],
    targets: Tensor | dict[str, Tensor],
    criterion: CombinedLoss,
) -> tuple[Tensor, float, float]:
    if isinstance(outputs, dict):
        if not isinstance(targets, dict):
            raise TypeError("targets must be dict for unidirectional training")

        head_losses: list[Tensor] = []
        head_dice: list[float] = []
        head_iou: list[float] = []
        for head in ("ant", "ret"):
            head_losses.append(criterion(outputs[head], targets[head]))
            head_dice.append(_dice_score(outputs[head], targets[head]))
            head_iou.append(_iou_score(outputs[head], targets[head]))

        loss = torch.stack(head_losses).mean()
        return loss, sum(head_iou) / len(head_iou), sum(head_dice) / len(head_dice)

    if isinstance(targets, dict):
        raise TypeError("targets cannot be dict for bidirectional training")

    loss = criterion(outputs, targets)
    return loss, _iou_score(outputs, targets), _dice_score(outputs, targets)


def _to_device(batch: tuple[Any, Any], device: torch.device) -> tuple[Any, Any]:
    x, y = batch
    x = x.to(device)
    if isinstance(y, dict):
        y = {k: v.to(device) for k, v in y.items()}
    else:
        y = y.to(device)
    return x, y


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_batches = 0

    iterator = tqdm(loader, desc=desc, leave=False)
    for batch in iterator:
        x, y = _to_device(batch, device)

        with torch.set_grad_enabled(is_train):
            outputs = model(x)
            loss, iou, dice = _compute_loss_and_metrics(outputs, y, criterion)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_iou += iou
        total_dice += dice
        total_batches += 1

        iterator.set_postfix(
            loss=f"{total_loss / total_batches:.4f}",
            iou=f"{total_iou / total_batches:.4f}",
            dice=f"{total_dice / total_batches:.4f}",
        )

    if total_batches == 0:
        raise RuntimeError("no batches processed")

    return {
        "loss": total_loss / total_batches,
        "iou": total_iou / total_batches,
        "dice": total_dice / total_batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KymoButler segmentation model from scratch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--mode", choices=["bidirectional", "unidirectional"], default="bidirectional")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "kymobutler")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    train_generator = SyntheticKymographGenerator(
        height=args.image_size,
        width=args.image_size,
        mode=args.mode,
        seed=args.seed,
    )
    val_generator = SyntheticKymographGenerator(
        height=args.image_size,
        width=args.image_size,
        mode=args.mode,
        seed=args.seed + 1,
    )

    train_ds = KymographDataset(num_samples=args.train_samples, generator=train_generator)
    val_ds = KymographDataset(num_samples=args.val_samples, generator=val_generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.mode == "bidirectional":
        model = build_unet(base_channels=args.base_channels)
    else:
        model = build_unet_unidirectional(base_channels=args.base_channels)
    model.to(device)

    criterion = CombinedLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            desc=f"train {epoch}/{args.epochs}",
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            desc=f"val {epoch}/{args.epochs}",
        )

        scheduler.step()

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/iou", train_metrics["iou"], epoch)
        writer.add_scalar("train/dice", train_metrics["dice"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/iou", val_metrics["iou"], epoch)
        writer.add_scalar("val/dice", val_metrics["dice"], epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        ckpt_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(ckpt_payload, ckpt_dir / "last.pt")

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(ckpt_payload, ckpt_dir / "best.pt")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_dice={val_metrics['dice']:.4f}"
        )

    writer.close()


if __name__ == "__main__":
    main()
