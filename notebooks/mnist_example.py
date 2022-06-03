from typing import Optional, Any
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        shuffle: Optional[bool] = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def _make_dataloader(self, dataset, stage: str):
        return DataLoader(
            Subset(dataset, list(range(100))),
            batch_size=self.batch_size,
            shuffle=self.shuffle if stage == "train" else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self):
        return self._make_dataloader(self.mnist_train, "train")

    def val_dataloader(self):
        return
    #     return self._make_dataloader(self.mnist_val, "val")

    def test_dataloader(self):
        return self._make_dataloader(self.mnist_test, "test")

    def predict_dataloader(self):
        return self._make_dataloader(self.mnist_predict, "predict")


class MNISTModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.Dropout(),
            nn.Linear(128, 10),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def step(self, batch: Any, stage: str) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log(f"{stage}/loss", loss)
        return {"loss": loss, "logits": logits}

    def training_step(self, batch, *args, **kwargs) -> Tensor:
        return self.step(batch, "train")

    def validation_step(self, batch, *args, **kwargs) -> Tensor:
        return self.step(batch, "val")

    def test_step(self, batch, *args, **kwargs) -> Tensor:
        return self.step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)
