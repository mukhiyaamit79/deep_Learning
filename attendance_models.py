"""PyTorch model definitions shared by the notebook and local inference apps."""
from attendance_runtime import configure_runtime
configure_runtime()

import torch
from torch import nn


class AttendanceCNN(nn.Module):
    """Three two-convolution blocks; NCHW input; unnormalized class logits."""

    def __init__(self, num_classes, filters=(32, 64, 128), dropout=0.3, dense_units=128):
        super().__init__()
        if num_classes < 2 or len(filters) != 3:
            raise ValueError("CNN needs at least two classes and three filter widths.")
        self.model_config = {"num_classes": int(num_classes), "filters": list(filters),
                             "dropout": float(dropout), "dense_units": int(dense_units)}
        blocks, channels = [], 1
        for width in filters:
            blocks.extend([
                nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(dropout / 2),
            ])
            channels = width
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, dense_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, images):
        return self.head(self.features(images))


class RowRecurrentClassifier(nn.Module):
    """A genuine simple RNN or LSTM over spatial image rows, not video frames."""

    def __init__(self, kind, num_classes, input_features=128, units=64,
                 dropout=0.3, dense_units=128):
        super().__init__()
        if kind not in ("RNN", "LSTM"):
            raise ValueError("kind must be RNN or LSTM.")
        self.kind = kind
        recurrent = nn.RNN if kind == "RNN" else nn.LSTM
        self.recurrent = recurrent(input_size=input_features, hidden_size=units,
                                   num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(units, dense_units),
                                  nn.ReLU(), nn.Linear(dense_units, num_classes))

    def forward(self, rows):
        output, _ = self.recurrent(rows)
        return self.head(output[:, -1, :])


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def print_model_summary(model):
    """Print structure and count without a synthetic forward pass."""
    print(model)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Stored parameter device: {next(model.parameters()).device} (the active training device is printed by the runner)")


def choose_device():
    """Require CUDA; there is deliberately no CPU inference/training fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    raise RuntimeError(
        "CUDA is unavailable. Select your Anaconda base Python kernel, restart it, "
        "and run from the setup cell. CPU fallback is disabled.")
