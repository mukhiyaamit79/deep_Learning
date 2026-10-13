"""Reusable PyTorch training, augmentation, evaluation and synchronized timing.

Importing this module does not train a model or open a camera.
"""
from attendance_runtime import configure_runtime
configure_runtime()

import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from attendance_models import count_parameters


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Warn for operations lacking deterministic kernels; do not claim bitwise reproducibility.
    torch.use_deterministic_algorithms(True, warn_only=True)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def prepare_inputs(images, device, sequence=False):
    """Convert shared NHWC arrays into NCHW CNN or NHW row-sequence tensors."""
    tensor = torch.as_tensor(images, dtype=torch.float32, device=device)
    if tensor.ndim != 4 or tensor.shape[-1] != 1:
        raise ValueError("Expected NHWC grayscale images with a singleton last axis.")
    return tensor[..., 0].contiguous() if sequence else tensor.permute(0, 3, 1, 2).contiguous()


class TrainingAugmentation:
    """Seeded per-image GPU augmentation, shared across every architecture.

    Horizontal flip; rotation +/-21.6 degrees; translation +/-8% per axis;
    independent axis zoom factors 0.9..1.1; additive brightness +/-0.15.
    Spatial transforms are combined into one bilinear warp with border padding.
    This is comparable in policy, not pixel-identical to the old Keras pipeline.
    """

    def __init__(self, device, seed=42):
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def __call__(self, images):
        # Input and output are NCHW. Independent RNG isolates augmentation from model dropout.
        batch_size = images.shape[0]
        values = torch.rand((batch_size, 7), device=images.device, generator=self.generator)
        angle = (values[:, 0] * 2 - 1) * (0.06 * 2 * torch.pi)
        scale_x, scale_y = 0.9 + values[:, 1] * 0.2, 0.9 + values[:, 2] * 0.2
        cosine, sine = torch.cos(angle), torch.sin(angle)
        theta = images.new_zeros((batch_size, 2, 3))
        theta[:, 0, 0], theta[:, 0, 1] = cosine * scale_x, -sine * scale_y
        theta[:, 1, 0], theta[:, 1, 1] = sine * scale_x, cosine * scale_y
        theta[:, :, 2] = (values[:, 3:5] * 2 - 1) * 0.16
        flipped = torch.where((values[:, 5] < 0.5)[:, None, None, None], images.flip(-1), images)
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        warped = F.grid_sample(flipped, grid, mode="bilinear", padding_mode="border", align_corners=False)
        brightness = (values[:, 6] * 2 - 1)[:, None, None, None] * 0.15
        return (warped + brightness).clamp(0, 1)


def make_loader(features, labels, batch_size=32, shuffle=False, seed=42, pin_memory=False):
    """Keep source tensors on CPU; one-process loading works in Windows/Jupyter."""
    images = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).permute(0, 3, 1, 2)
    targets = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
    return DataLoader(TensorDataset(images, targets), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0,
                      pin_memory=pin_memory, drop_last=False)


def run_epoch(model, loader, device, sequence=False, optimizer=None, augmentation=None,
              collect_predictions=False):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, seen = 0.0, 0, 0
    probabilities = []
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training and augmentation is not None:
                images = augmentation(images)
            inputs = images.squeeze(1) if sequence else images
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stop and inspect data/learning rate before testing.")
            if training:
                loss.backward()
                optimizer.step()
            count = targets.size(0)
            loss_sum += loss.detach().item() * count
            correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            seen += count
            if collect_predictions:
                probabilities.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
    if not seen:
        raise ValueError("Cannot train/evaluate an empty split.")
    return {"loss": loss_sum / seen, "accuracy": correct / seen,
            "probabilities": np.concatenate(probabilities) if collect_predictions else None}


def fit_model(model, train_loader, val_loader, device, epochs=120, patience=25,
              learning_rate=1e-3, sequence=False, seed=42, verbose=2):
    """Train on device, restore minimum-val-loss weights, then retain the model on CPU.

    Keeping completed experiments on CPU avoids filling the laptop GPU with all runs.
    No automatic batch-size or device changes are made after a memory failure.
    """
    model.to(device)
    if next(model.parameters()).device.type != torch.device(device).type:
        raise RuntimeError("Model was not placed on the selected training device.")
    print(f"Training parameter device: {next(model.parameters()).device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, threshold=0.0,
        threshold_mode="abs", min_lr=1e-6)
    augmentation = TrainingAugmentation(device, seed) if getattr(train_loader, "apply_augmentation", True) else None
    history = {key: [] for key in ("loss", "accuracy", "val_loss", "val_accuracy", "learning_rate")}
    best_loss, best_state, best_index, stale_epochs = float("inf"), None, -1, 0
    synchronize(device)
    start = time.perf_counter()
    try:
        for epoch in range(epochs):
            train = run_epoch(model, train_loader, device, sequence, optimizer, augmentation)
            val = run_epoch(model, val_loader, device, sequence)
            history["loss"].append(train["loss"])
            history["accuracy"].append(train["accuracy"])
            history["val_loss"].append(val["loss"])
            history["val_accuracy"].append(val["accuracy"])
            history["learning_rate"].append(optimizer.param_groups[0]["lr"])
            if val["loss"] < best_loss:
                best_loss, best_index, stale_epochs = val["loss"], epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale_epochs += 1
            scheduler.step(val["loss"])
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | loss {train['loss']:.4f} "
                      f"acc {train['accuracy']:.4f} | val_loss {val['loss']:.4f} "
                      f"val_acc {val['accuracy']:.4f} | lr {optimizer.param_groups[0]['lr']:.2g}")
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}; restoring epoch {best_index+1}.")
                break
        synchronize(device)
        elapsed = time.perf_counter() - start
        if best_state is None:
            raise RuntimeError("No valid epoch completed.")
        model.load_state_dict(best_state)
        restored = run_epoch(model, val_loader, device, sequence)
        return {"history": history, "seconds": elapsed, "best_index": best_index,
                "restored": restored, "parameters": count_parameters(model)}
    finally:
        model.cpu().eval()
        # The optimizer is local and is released on return; no completed run retains GPU state.


def evaluate_model(model, features, labels, device, sequence=False, batch_size=32):
    loader = make_loader(features, labels, batch_size=batch_size,
                         pin_memory=torch.device(device).type == "cuda")
    model.to(device)
    try:
        return run_epoch(model, loader, device, sequence=sequence, collect_predictions=True)
    finally:
        model.cpu().eval()


def benchmark_model(model, images, device, sequence=False, repetitions=20, warmups=3, batch_size=32):
    """Synchronized model latency; input transfer/preprocessing excluded for all models."""
    timings = {}
    batch_size = min(batch_size, len(images))
    model.to(device).eval()
    try:
        with torch.inference_mode():
            for size, label in ((1, "single"), (batch_size, "batch")):
                inputs = prepare_inputs(images[:size], device, sequence)
                for _ in range(warmups):
                    torch.softmax(model(inputs), dim=1)
                synchronize(device)
                durations = []
                for _ in range(repetitions):
                    synchronize(device)
                    start = time.perf_counter()
                    torch.softmax(model(inputs), dim=1)
                    synchronize(device)
                    durations.append(time.perf_counter() - start)
                timings[label] = float(np.median(durations))
    finally:
        model.cpu().eval()
    return {"Inference Time (ms/image)": timings["single"] * 1000,
            "Batch Time (ms)": timings["batch"] * 1000,
            "Batch Throughput (images/s)": batch_size / timings["batch"]}
