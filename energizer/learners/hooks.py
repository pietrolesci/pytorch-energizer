"""Forward hooks to underlying Lightning module."""

from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.optimizer import Optimizer


class ModelHooks:
    """Hooks to be used in LightningModule."""

    #############
    # START/END #
    #############
    def on_fit_start(self) -> None:
        return LightningModule.on_fit_start(self.learner)

    def on_fit_end(self) -> None:
        return LightningModule.on_fit_end(self.learner)

    def on_train_start(self) -> None:
        return LightningModule.on_train_start(self.learner)

    def on_train_end(self) -> None:
        return LightningModule.on_train_end(self.learner)

    def on_validation_start(self) -> None:
        return LightningModule.on_validation_start(self.learner)

    def on_validation_end(self) -> None:
        return LightningModule.on_validation_end(self.learner)

    def on_test_start(self) -> None:
        return LightningModule.on_test_start(self.learner)

    def on_test_end(self) -> None:
        return LightningModule.on_test_end(self.learner)

    def on_pool_start(self) -> None:
        pass

    def on_pool_end(self) -> None:
        pass

    def on_predict_start(self) -> None:
        return LightningModule.on_predict_start(self.learner)

    def on_predict_end(self) -> None:
        return LightningModule.on_predict_end(self.learner)

    ###################
    # BATCH START/END #
    ###################
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        return LightningModule.on_train_batch_start(self.learner, batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        return LightningModule.on_train_batch_end(self.learner, outputs, batch, batch_idx)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return LightningModule.on_validation_batch_start(self.learner, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        return LightningModule.on_validation_batch_end(self.learner, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return LightningModule.on_test_batch_start(self.learner, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        return LightningModule.on_test_batch_end(self.learner, outputs, batch, batch_idx, dataloader_idx)

    def on_pool_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        pass

    def on_pool_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        pass

    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return LightningModule.on_predict_batch_start(self.learner, batch, batch_idx, dataloader_idx)

    def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return LightningModule.on_predict_batch_end(self.learner, outputs, batch, batch_idx, dataloader_idx)

    ##############
    # MODEL EVAL #
    ##############
    def on_validation_model_eval(self) -> None:
        return LightningModule.on_validation_model_eval(self.learner)

    def on_validation_model_train(self) -> None:
        return LightningModule.on_validation_model_train(self.learner)

    def on_test_model_train(self) -> None:
        return LightningModule.on_test_model_train(self.learner)

    def on_test_model_eval(self) -> None:
        return LightningModule.on_test_model_eval(self.learner)

    def on_pool_model_train(self) -> None:
        pass

    def on_pool_model_eval(self) -> None:
        pass

    def on_predict_model_eval(self) -> None:
        return LightningModule.on_predict_model_eval(self.learner)

    ###################
    # EPOCH START/END #
    ###################
    def on_train_epoch_start(self) -> None:
        return LightningModule.on_train_epoch_start(self.learner)

    def on_train_epoch_end(self) -> None:
        return LightningModule.on_train_epoch_end(self.learner)

    def on_validation_epoch_start(self) -> None:
        return LightningModule.on_validation_epoch_start(self.learner)

    def on_validation_epoch_end(self) -> None:
        return LightningModule.on_validation_epoch_end(self.learner)

    def on_test_epoch_start(self) -> None:
        return LightningModule.on_test_epoch_start(self.learner)

    def on_test_epoch_end(self) -> None:
        return LightningModule.on_test_epoch_end(self.learner)

    def on_pool_epoch_start(self) -> None:
        pass

    def on_pool_epoch_end(self) -> None:
        pass

    def on_predict_epoch_start(self) -> None:
        return LightningModule.on_predict_epoch_start(self.learner)

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        return LightningModule.on_predict_epoch_end(self.learner, results)

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        return LightningModule.on_before_zero_grad(self.learner, optimizer)

    def on_before_backward(self, loss: torch.Tensor) -> None:
        return LightningModule.on_before_backward(self.learner, loss)

    def on_after_backward(self) -> None:
        return LightningModule.on_after_backward(self.learner)

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        return LightningModule.on_before_optimizer_step(self.learner, optimizer, optimizer_idx)

    def configure_sharded_model(self) -> None:
        return LightningModule.configure_sharded_model(self.learner)


class DataHooks:
    """Hooks to be used for data related stuff."""

    def prepare_data(self) -> None:
        return LightningModule.prepare_data(self.learner)

    def setup(self, stage: Optional[str] = None) -> None:
        return LightningModule.setup(self.learner, stage)

    def teardown(self, stage: Optional[str] = None) -> None:
        return LightningModule.teardown(self.learner, stage)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        return LightningModule.transfer_batch_to_device(self.learner, batch, device, dataloader_idx)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return LightningModule.on_before_batch_transfer(self.learner, batch, dataloader_idx)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return LightningModule.on_after_batch_transfer(self.learner, batch, dataloader_idx)

    def on_pool_dataloader(self, *_: Any):
        """To deprecate when PL deprecates the similar ones.

        This is used by PL in trainer._data_connector._request_dataloader
        """


class CheckpointHooks:
    """Hooks to be used with Checkpointing."""

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        return LightningModule.on_load_checkpoint(self.learner, checkpoint)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        return LightningModule.on_save_checkpoint(self.learner, checkpoint)
