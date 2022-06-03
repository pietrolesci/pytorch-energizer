import importlib
from typing import Any, List, Optional, Union

from attr import has
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar, _update_n, convert_inf
from pytorch_lightning.utilities.types import STEP_OUTPUT

if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

import sys


class TQDMProgressBarPool(TQDMProgressBar):
    # TODO: this is not working I can still see Testing
    @property
    def test_description(self) -> str:
        if hasattr(self.trainer.datamodule, "is_on_pool") and self.trainer.datamodule.is_on_pool:
            return "Pool Evaluation"
        return super().test_description


# class TQDMProgressBarPool(TQDMProgressBar):
#     # TODO: this is not working I can still see Testing

#     def __init__(self, refresh_rate: int = 1, process_position: int = 0):
#         super().__init__(refresh_rate, process_position)
#         self._pool_progress_bar: Optional[_tqdm] = None

#     @property
#     def pool_description(self) -> str:
#         return "Pool"

#     @property
#     def pool_batch_idx(self) -> int:
#         """The number of batches processed during testing.

#         Use this to update your progress bar.
#         """
#         return self.trainer.active_learning_loop.pool_loop.epoch_loop.batch_progress.current.processed

#     @property
#     def total_pool_batches_current_dataloader(self) -> Union[int, float]:
#         """The total number of testing batches, which may change from epoch to epoch for current dataloader.

#         Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the test dataloader is
#         of infinite size.
#         """
#         # NOTE: since we are using the flag for the dataloader instead of adding a `pool_dataloader` method
#         return super().total_test_batches_current_dataloader

#     @property
#     def pool_progress_bar(self) -> _tqdm:
#         if self._pool_progress_bar is None:
#             raise TypeError(f"The `{self.__class__.__name__}._pool_progress_bar` reference has not been set yet.")
#         return self._pool_progress_bar

#     @pool_progress_bar.setter
#     def pool_progress_bar(self, bar: _tqdm) -> None:
#         self._pool_progress_bar = bar

#     def init_pool_tqdm(self) -> Tqdm:
#         """Override this to customize the tqdm bar for testing."""
#         bar = Tqdm(
#             desc=self.pool_description,
#             position=(2 * self.process_position),
#             disable=self.is_disabled,
#             leave=True,
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )
#         return bar

#     def on_pool_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         self.pool_progress_bar = self.init_pool_tqdm()

#     def on_pool_batch_start(
#         self,
#         trainer: Trainer,
#         pl_module: LightningModule,
#         batch: Any,
#         batch_idx: int,
#         dataloader_idx: int,
#     ) -> None:
#         if self.has_dataloader_changed(dataloader_idx):
#             return
#         # self._current_eval_dataloader_idx = dataloader_idx or 0
#         self.pool_progress_bar.reset(convert_inf(self.total_pool_batches_current_dataloader))
#         self.pool_progress_bar.set_description(f"{self.pool_description} DataLoader {dataloader_idx}")
#         print("ERROR")

#     def on_pool_batch_end(self, *_: Any) -> None:
#         if self._should_update(self.pool_batch_idx, self.pool_progress_bar.total):
#             print(self.pool_batch_idx)
#             _update_n(self.pool_progress_bar, self.pool_batch_idx, self.refresh_rate)

#     def on_pool_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         self.pool_progress_bar.close()
#         self.reset_dataloader_idx_tracker()

#     def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
#         active_progress_bar = None

#         if self._main_progress_bar is not None and not self.main_progress_bar.disable:
#             active_progress_bar = self.main_progress_bar
#         elif self._val_progress_bar is not None and not self.val_progress_bar.disable:
#             active_progress_bar = self.val_progress_bar
#         elif self._test_progress_bar is not None and not self.test_progress_bar.disable:
#             active_progress_bar = self.test_progress_bar
#         elif self._pool_progress_bar is not None and not self.pool_progress_bar.disable:
#             active_progress_bar = self.pool_progress_bar
#         elif self._predict_progress_bar is not None and not self.predict_progress_bar.disable:
#             active_progress_bar = self.predict_progress_bar

#         if active_progress_bar is not None:
#             s = sep.join(map(str, args))
#             active_progress_bar.write(s, **kwargs)


class CallBackPoolHooks:
    @staticmethod
    def on_pool_batch_start(
        trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the test batch begins."""

    @staticmethod
    def on_pool_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> None:
        """Called when the test batch ends."""

    @staticmethod
    def on_pool_epoch_start(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch begins."""

    @staticmethod
    def on_pool_epoch_end(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch ends."""

    @staticmethod
    def on_pool_start(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test begins."""

    @staticmethod
    def on_pool_end(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test ends."""

    @staticmethod
    def _add_pool_hooks(callback: Callback) -> Callback:
        if not hasattr(callback, "on_pool_batch_start"):
            callback.on_pool_batch_start = CallBackPoolHooks.on_pool_batch_start
        if not hasattr(callback, "on_pool_batch_end"):
            callback.on_pool_batch_end = CallBackPoolHooks.on_pool_batch_end
        if not hasattr(callback, "on_pool_epoch_start"):
            callback.on_pool_epoch_start = CallBackPoolHooks.on_pool_epoch_start
        if not hasattr(callback, "on_pool_epoch_end"):
            callback.on_pool_epoch_end = CallBackPoolHooks.on_pool_epoch_end
        if not hasattr(callback, "on_pool_start"):
            callback.on_pool_start = CallBackPoolHooks.on_pool_start
        if not hasattr(callback, "on_pool_end"):
            callback.on_pool_end = CallBackPoolHooks.on_pool_end

        return callback


def patch_callbacks(callbacks: List[Callback]) -> List[Callback]:
    new_callbacks = []
    for c in callbacks:
        if isinstance(c, ProgressBarBase):
            prog_bar = TQDMProgressBarPool(process_position=c.process_position, refresh_rate=c.refresh_rate)
            prog_bar = CallBackPoolHooks._add_pool_hooks(prog_bar)
            new_callbacks.append(prog_bar)
        else:
            new_callbacks.append(CallBackPoolHooks._add_pool_hooks(c))

    return new_callbacks


# class TQDMProgressBarPool(TQDMProgressBar):
#     # TODO: this is not working I can still see Testing

#     def __init__(self, refresh_rate: int = 1, process_position: int = 0):
#         super().__init__(refresh_rate, process_position)
#         self._general_progress_bar: Optional[_tqdm] = None
#         self._pool_progress_bar: Optional[_tqdm] = None

#     @property
#     def general_progress_bar(self) -> _tqdm:
#         if self._general_progress_bar is None:
#             raise TypeError(f"The `{self.__class__.__name__}._val_progress_bar` reference has not been set yet.")
#         return self._general_progress_bar

#     @general_progress_bar.setter
#     def general_progress_bar(self, bar: _tqdm) -> None:
#         self._general_progress_bar = bar

#     @property
#     def pool_progress_bar(self) -> _tqdm:
#         if self._pool_progress_bar is None:
#             raise TypeError(f"The `{self.__class__.__name__}._pool_progress_bar` reference has not been set yet.")
#         return self._pool_progress_bar

#     @pool_progress_bar.setter
#     def pool_progress_bar(self, bar: _tqdm) -> None:
#         self._pool_progress_bar = bar

#     @property
#     def general_description(self) -> str:
#         return "Labelling Iter"

#     @property
#     def pool_description(self) -> str:
#         return "Pool"

#     @property
#     def general_batch_idx(self) -> int:
#         """The number of batches processed during training.

#         Use this to update your progress bar.
#         """
#         return self.trainer.active_learning_loop.epoch_progress.current.processed

#     @property
#     def pool_batch_idx(self) -> int:
#         """The number of batches processed during testing.

#         Use this to update your progress bar.
#         """
#         return self.trainer.active_learning_loop.pool_loop.epoch_loop.batch_progress.current.processed

#     @property
#     def total_pool_batches_current_dataloader(self) -> Union[int, float]:
#         """The total number of testing batches, which may change from epoch to epoch for current dataloader.

#         Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the test dataloader is
#         of infinite size.
#         """
#         # NOTE: since we are using the flag for the dataloader instead of adding a `pool_dataloader` method
#         return super().total_test_batches_current_dataloader

#     def init_general_tqdm(self) -> Tqdm:
#         """Override this to customize the tqdm bar for training."""
#         bar = Tqdm(
#             desc=self.main_description,
#             initial=self.main_batch_idx,
#             position=(2 * self.process_position),
#             disable=self.is_disabled,
#             leave=True,
#             dynamic_ncols=True,
#             file=sys.stdout,
#             smoothing=0,
#         )
#         return bar

#     def init_pool_tqdm(self) -> Tqdm:
#         """Override this to customize the tqdm bar for testing."""
#         bar = Tqdm(
#             desc=self.pool_description,
#             position=(2 * self.process_position),
#             disable=self.is_disabled,
#             leave=True,
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )
#         return bar

#     def on_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         self.general_progress_bar = self.init_general_tqdm()

#         # total fit batches
#         total_train_batches = self.total_train_batches
#         total_val_batches = self.total_val_batches
#         if total_train_batches != float("inf") and total_val_batches != float("inf"):
#             # val can be checked multiple times per epoch
#             val_checks_per_epoch = total_train_batches // trainer.val_check_batch
#             total_val_batches = total_val_batches * val_checks_per_epoch
#         total_fit_batches = total_train_batches + total_val_batches

#         # pool batches


#     def on_pool_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         self.pool_progress_bar = self.init_pool_tqdm()

#     def on_pool_batch_start(
#         self,
#         trainer: Trainer,
#         pl_module: LightningModule,
#         batch: Any,
#         batch_idx: int,
#         dataloader_idx: int,
#     ) -> None:
#         if self.has_dataloader_changed(dataloader_idx):
#             return
#         # self._current_eval_dataloader_idx = dataloader_idx or 0
#         self.pool_progress_bar.reset(convert_inf(self.total_pool_batches_current_dataloader))
#         self.pool_progress_bar.set_description(f"{self.pool_description} DataLoader {dataloader_idx}")
#         print("ERROR")

#     def on_pool_batch_end(self, *_: Any) -> None:
#         if self._should_update(self.pool_batch_idx, self.pool_progress_bar.total):
#             print(self.pool_batch_idx)
#             _update_n(self.pool_progress_bar, self.pool_batch_idx, self.refresh_rate)

#     def on_pool_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         self.pool_progress_bar.close()
#         self.reset_dataloader_idx_tracker()

#     def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
#         active_progress_bar = None

#         if self._main_progress_bar is not None and not self.main_progress_bar.disable:
#             active_progress_bar = self.main_progress_bar
#         elif self._val_progress_bar is not None and not self.val_progress_bar.disable:
#             active_progress_bar = self.val_progress_bar
#         elif self._test_progress_bar is not None and not self.test_progress_bar.disable:
#             active_progress_bar = self.test_progress_bar
#         elif self._pool_progress_bar is not None and not self.pool_progress_bar.disable:
#             active_progress_bar = self.pool_progress_bar
#         elif self._predict_progress_bar is not None and not self.predict_progress_bar.disable:
#             active_progress_bar = self.predict_progress_bar

#         if active_progress_bar is not None:
#             s = sep.join(map(str, args))
#             active_progress_bar.write(s, **kwargs)
