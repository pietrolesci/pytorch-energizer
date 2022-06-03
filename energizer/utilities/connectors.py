from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.trainer.connectors.data_connector import DataConnector as DataConnector_pl
from pytorch_lightning.trainer.connectors.data_connector import _DataLoaderSource
from pytorch_lightning.utilities import LightningEnum

from energizer.data.datamodule import ActiveDataModule


class PoolRunningStage(LightningEnum):
    POOL = "pool"

    @property
    def evaluating(self) -> bool:
        return True

    @property
    def dataloader_prefix(self) -> Optional[str]:
        return self.value


class DataConnector(DataConnector_pl):
    def __init__(self, trainer: "pl.Trainer", multiple_trainloader_mode: str = "max_size_cycle"):
        super().__init__(trainer, multiple_trainloader_mode)
        self._pool_dataloader_source = _DataLoaderSource(None, "")

    def attach_datamodule(
        self, model: "pl.LightningModule", datamodule: Optional["pl.LightningDataModule"] = None
    ) -> None:
        # do the usual
        super().attach_datamodule(model, datamodule)

        # and attach the pool dataloader if the user has passed an ActiveDataModule
        if isinstance(datamodule, ActiveDataModule):
            self._pool_dataloader_source = _DataLoaderSource(datamodule, "pool_dataloader")
