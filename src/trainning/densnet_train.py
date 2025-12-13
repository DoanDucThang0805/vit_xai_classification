from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from .trainer import Trainer
from dataset.dataset import train_dataset, val_dataset
from model.densnet121 import model


BATCH_SIZE = 32
train_ds = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_ds = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = Path.cwd().parents[0]

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)

trainer = Trainer(
    num_epochs=200,
    device=device,
    batch_size=BATCH_SIZE,
    train_loader=train_ds,
    val_loader=val_ds,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    checkpoints_dir=str(output_dir / "checkpoints" / "plantdoc" / "densnet121")
)

if __name__ == "__main__":
    trainer.train()
