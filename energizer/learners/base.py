from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn

from energizer.learners.hooks import CheckpointHooks, DataHooks, ModelHooks
from energizer.utilities.learners import ModuleWrapperBase, patch_dropout_layers


class Learner(ModuleWrapperBase, ModelHooks, DataHooks, CheckpointHooks, LightningModule):
    def __init__(self, learner: LightningModule):
        super().__init__()
        self.learner = learner

    def forward(self, *args, **kwargs) -> Any:
        return self.learner(*args, **kwargs)

    def training_step(self, *args, **kwargs) -> Any:
        return self.learner.training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> None:
        return LightningModule.validation_step(self.learner, *args, **kwargs)

    def test_step(self, *args, **kwargs) -> None:
        return LightningModule.test_step(self.learner, *args, **kwargs)

    def training_epoch_end(self, outputs: Any) -> None:
        return LightningModule.training_epoch_end(self.learner, outputs)

    def validation_epoch_end(self, outputs: Any) -> None:
        return LightningModule.validation_epoch_end(self.learner, outputs)

    def test_epoch_end(self, outputs: Any) -> None:
        return LightningModule.test_epoch_end(self.learner, outputs)

    def pool_step(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def pool_step_end(self, *args, **kwargs) -> Any:
        pass

    def pool_epoch_end(self, *args, **kwargs) -> Optional[Any]:
        pass

    def configure_optimizers(self):
        return self.learner.configure_optimizers()


class Deterministic(Learner):
    pass


class MCDropout(Learner):
    """Implements the MCDropout inference method in [PUT REFERENCES]."""

    def __init__(
        self,
        learner: nn.Module,
        query_size: int,
        num_inference_iters: int = 10,
        consistent: bool = False,
        inplace: bool = False,
        prob: Optional[float] = None,
    ) -> None:
        """Instantiates a new learner (same as `learner`) with patched dropout layers.

        The patched such that they are active even during evaluation.

        Args:
            num_inference_iters (int): The number of forward passes to perform.
            consistent (bool): If True, it uses the consistent version of dropout that fixes the mask across batches.
            prob (float): If specified, this changes the dropout probability of all layers to `prob`. If `None` the
                dropout probability is the same as the original layer. Must be 0 <= prob <= 1.
            inplace (bool): Whether to modify the learner in place or return a copy of the learner.
        """
        self.num_inference_iters = num_inference_iters
        self.consistent = consistent
        self.prob = prob
        self.inplace = inplace
        learner = patch_dropout_layers(
            module=learner,
            prob=prob,
            consistent=consistent,
            inplace=inplace,
        )
        super().__init__(learner, query_size)

    def forward(self, *args, **kwargs) -> Tensor:
        """Performs `num_inference_iters` forward passes using the underlying learner and keeping the dropout layers active..

        Returns:
            A tensor of dimension `(B: batch_size, C: num_classes, S: num_samples)`.
        """
        out = []
        for _ in range(self.num_inference_iters):
            out.append(self.learner(*args, **kwargs))  # type: ignore
        # expects a shape [num_samples, num_classes, num_iterations]
        return torch.stack(out).permute((1, 2, 0))
