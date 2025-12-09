import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HairNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 3x200x200 -> Conv -> 32x198x198 -> Pool -> 32x99x99
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 99 * 99, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_loaders(
    data_dir: Path,
    train_tfms: transforms.Compose,
    test_tfms: transforms.Compose,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    train_ds = datasets.ImageFolder(root=data_dir / "train", transform=train_tfms)
    test_ds = datasets.ImageFolder(root=data_dir / "test", transform=test_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return {"train": train_loader, "test": test_loader, "train_ds": train_ds, "test_ds": test_ds}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    model.train() if train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return {"loss": epoch_loss, "acc": epoch_acc}


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    num_epochs: int,
    history: Dict[str, List[float]],
    start_epoch: int = 0,
    tag: str = "",
) -> Dict[str, List[float]]:
    for epoch in range(num_epochs):
        train_metrics = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, loaders["test"], criterion, optimizer, device, train=False)

        history.setdefault("loss", []).append(train_metrics["loss"])
        history.setdefault("acc", []).append(train_metrics["acc"])
        history.setdefault("val_loss", []).append(val_metrics["loss"])
        history.setdefault("val_acc", []).append(val_metrics["acc"])

        print(
            f"{tag}Epoch {start_epoch + epoch + 1}/{start_epoch + num_epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f}"
        )
    return history


def compute_summary(history: Dict[str, List[float]], first_phase: int, second_phase: int) -> Dict[str, float]:
    median_train_acc_first = float(np.median(history["acc"][:first_phase]))
    std_train_loss_first = float(np.std(history["loss"][:first_phase]))
    mean_val_loss_aug = float(np.mean(history["val_loss"][first_phase:first_phase + second_phase]))
    avg_val_acc_last5_aug = float(np.mean(history["val_acc"][first_phase + second_phase - 5:first_phase + second_phase]))

    return {
        "median_train_acc_first_phase": median_train_acc_first,
        "std_train_loss_first_phase": std_train_loss_first,
        "mean_val_loss_aug_phase": mean_val_loss_aug,
        "avg_val_acc_last5_aug_phase": avg_val_acc_last5_aug,
    }


def main():
    parser = argparse.ArgumentParser(description="Hair Type CNN Training (ML Zoomcamp Homework)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to dataset root containing train/ and test/")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=10, help="Epochs for initial training")
    parser.add_argument("--aug-epochs", type=int, default=10, help="Additional epochs with augmentation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("hair_type_homework/model.pt"), help="Where to save model weights")
    parser.add_argument("--results", type=Path, default=Path("hair_type_homework/results.json"), help="Where to save metrics summary")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_transforms = transforms.Compose(
        [
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    aug_transforms = transforms.Compose(
        [
            transforms.RandomRotation(50),
            transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    loaders_phase1 = build_loaders(args.data_dir, base_transforms, base_transforms, args.batch_size, args.num_workers)

    model = HairNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

    history: Dict[str, List[float]] = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    history = train_model(
        model,
        criterion,
        optimizer,
        loaders_phase1,
        device,
        num_epochs=args.num_epochs,
        history=history,
        start_epoch=0,
        tag="[Base] ",
    )

    loaders_phase2 = build_loaders(args.data_dir, aug_transforms, base_transforms, args.batch_size, args.num_workers)
    history = train_model(
        model,
        criterion,
        optimizer,
        loaders_phase2,
        device,
        num_epochs=args.aug_epochs,
        history=history,
        start_epoch=args.num_epochs,
        tag="[Aug ] ",
    )

    summary = compute_summary(history, first_phase=args.num_epochs, second_phase=args.aug_epochs)
    print("\nSummary metrics:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"model_state_dict": model.state_dict(), "history": history}, args.checkpoint)

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump({"history": history, "summary": summary}, f, indent=2)

    print(f"\nSaved checkpoint to {args.checkpoint}")
    print(f"Saved metrics to {args.results}")


if __name__ == "__main__":
    main()
