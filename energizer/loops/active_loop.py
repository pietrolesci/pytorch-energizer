from copy import deepcopy
from typing import Any, List, Optional

import torch
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.utilities import _is_max_limit_reached
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from energizer.loops.pool_loop import PoolEvaluationLoop
from energizer.trainer import Trainer


class ActiveLearningLoop(Loop):
    def __init__(
        self,
        reset_weights: bool,
        test_after_labelling: bool,
        total_budget: int,
        min_labelling_iters: int,
        max_labelling_iters: int,
    ):
        # super().__init__(min_epochs=min_labelling_iters, max_epochs=max_labelling_iters)
        super().__init__()
        self.min_labelling_iters = min_labelling_iters
        self.max_labelling_iters = max_labelling_iters
        self.reset_weights = reset_weights
        self.total_budget = total_budget
        self.test_after_labelling = test_after_labelling
        self.epoch_progress = Progress()

        # pool evaluation loop
        self.pool_loop: Optional[PoolEvaluationLoop] = None

        # fit after each labelling session
        self.active_fit_loop: Optional[FitLoop] = None

        # test after labelling and fitting
        self.active_test_loop: Optional[EvaluationLoop] = None

    def connect(
        self, pool_loop: PoolEvaluationLoop, active_fit_loop: FitLoop, active_test_loop: EvaluationLoop
    ) -> None:
        self.pool_loop = pool_loop
        self.active_fit_loop = active_fit_loop
        self.active_test_loop = active_test_loop

    def reset(self) -> None:
        """Resets the internal state of this loop."""
        if self.restarting:
            self.epoch_progress.reset_on_restart()

    @property
    def done(self) -> bool:
        """Evaluates when to leave the loop."""
        # `processed` is increased before `on_train_epoch_end`, the hook where checkpoints are typically saved.
        # we use it here because the checkpoint data won't have `completed` increased yet
        has_labelled_data = (
            not self.trainer.datamodule.has_unlabelled_data
            or self.trainer.query_size > self.trainer.datamodule.pool_size
        )
        stop_total_budget = _is_max_limit_reached(self.trainer.datamodule.labelled_size, self.total_budget)
        stop_epochs = _is_max_limit_reached(self.epoch_progress.current.processed, self.max_labelling_iters)
        if stop_epochs:
            # in case they are not equal, override so `trainer.current_epoch` has the expected value
            self.epoch_progress.current.completed = self.epoch_progress.current.processed

        self.trainer.should_stop = False

        return stop_epochs or has_labelled_data or stop_total_budget

    @property
    def skip(self) -> bool:
        """Whether we should skip the training and immediately return from the call to :meth:`run`."""
        # since `trainer.num_training_batches` depends on the `train_dataloader` but that won't be called
        # until `on_run_start`, we use `limit_train_batches` instead
        return self.done

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Store the original weights of the model."""
        # make a copy of the initial state of the model to reset the weights
        if self.reset_weights:
            self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

        self.epoch_progress.current.completed = self.epoch_progress.current.processed

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        self.epoch_progress.increment_ready()
        self.trainer._logger_connector.on_epoch_start()

        self.trainer._call_callback_hooks("on_epoch_start")
        self.trainer._call_lightning_module_hook("on_epoch_start")

        self.epoch_progress.increment_started()

        return super().on_advance_start(*args, **kwargs)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting, testing, and pool evaluation."""
        if self.trainer.datamodule.has_labelled_data:
            self._reset_fitting()  # requires to reset the tracking stage.
            self.active_fit_loop.run()
            self.active_fit_loop.epoch_progress.reset()
            # print(len(self.trainer.train_dataloader.dataset))
            # print(self.trainer.num_training_batches)

            if self.test_after_labelling:
                self._reset_testing()  # requires to reset the tracking stage.
                self.active_test_loop.run()

        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_pool()  # requires to reset the tracking stage.
            _, indices = self.pool_loop.run()
            self.labelling_loop(indices)

    def on_advance_end(self) -> None:
        """Used to restore the original weights and the optimizers and schedulers states."""
        self.trainer._logger_connector.epoch_end_reached()

        if self.reset_weights:
            self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
            self.trainer.accelerator.setup(self.trainer)  # TODO: do I need this to reset optimizers?

        self.epoch_progress.increment_processed()

        self.trainer._call_callback_hooks("on_epoch_end")
        self.trainer._call_lightning_module_hook("on_epoch_end")

        self.trainer._logger_connector.on_epoch_end()
        self.epoch_progress.increment_completed()

        # if fault tolerant is enabled and process has been notified, exit.
        self.trainer._exit_gracefully_on_signal()

    def on_run_end(self):
        # enforce training at the end of acquisitions
        self._reset_fitting()
        self.active_fit_loop.run()

    def teardown(self) -> None:
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True
        self.trainer.lightning_module.train()
        torch.set_grad_enabled(True)

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True
        self.trainer.lightning_module.eval()
        torch.set_grad_enabled(False)

    def _reset_pool(self) -> None:
        self.trainer.reset_pool_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True
        self.trainer.lightning_module.eval()
        torch.set_grad_enabled(False)

    @rank_zero_only
    def labelling_loop(self, indices: List[int]):
        self.trainer.datamodule.label(indices)
