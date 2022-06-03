import contextlib
import random
from copy import deepcopy
from itertools import cycle
from typing import Generator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import _collect_rng_states, _set_rng_states
from torch import Tensor, nn
from torch.nn.modules.dropout import _DropoutNd


class ModuleWrapperBase:
    """The ``ModuleWrapperBase`` is a base for classes which wrap a ``LightningModule`` or an instance of
    ``ModuleWrapperBase``.

    This class ensures that trainer attributes are forwarded to any wrapped or nested
    ``LightningModule`` instances so that nested calls to ``.log`` are handled correctly. The ``ModuleWrapperBase`` is
    also stateful. Attached state will be forwarded to any nested ``ModuleWrapperBase`` instances.

    Credits: Pytorch-Lightning Team.
    """

    def __init__(self):
        super().__init__()
        self._children = []

    def __setattr__(self, key, value):
        if isinstance(value, (LightningModule, ModuleWrapperBase)):
            self._children.append(key)
        patched_attributes = ["_current_fx_name", "_current_hook_fx_name", "_results", "_data_pipeline_state"]
        if isinstance(value, Trainer) or key in patched_attributes:
            if hasattr(self, "_children"):
                for child in self._children:
                    setattr(getattr(self, child), key, value)
        super().__setattr__(key, value)


@contextlib.contextmanager
def local_seed(seed: int) -> Generator[None, None, None]:
    """A context manager that allows to locally change the seed.

    Upon exit from the context manager it resets the random number generator state
    so that the operations that happen in the context do not affect randomness outside
    of it.
    """
    # collect current states
    states = _collect_rng_states()

    # set seed in the context
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # run code in context
    yield

    # reset states when exiting the context
    _set_rng_states(states)


class EnergizerDropoutLayer(_DropoutNd):
    """Base class for dropout layers that remains active even during evaluation.

    This class is used as a base to redefine all dropout layers present in Pytorch such
    that they remain always on even during evaluation. The elements to zero are randomized
    on every forward call during training time.

    This is obtained by overloading the original forward method of the class such that

    ```python
    # this
    F.dropout(input=input, p=self.p, training=self.training, inplace=self.inplace)

    # is turned into
    F.dropout(input=input, p=self.p, training=True, inplace=self.inplace)
    ```

    that is, the `training` argument is always set to `True`.

    Check the original documentation here: https://pytorch.org/docs/stable/nn.html#dropout-layers.

    Optionally, it is possible to guarantee that the dropout mask is consistent during validation
    and testing. That is, during evaluation time, a fixed mask is picked and kept until `reset_mask()`.
    This is useful to implement techniques like MCDropout that require multiple, say `S`, forward passes
    with the dropout layers active.
    During testing, this would create additional noise, due to the dropout layers still active. WIth
    consistent dropout, the same `S` masks are used (i.e., each one of the `S` forward passes will have
    the same mask unless it is manually reset).
    One of the advantages of the `Energizer` library is that it implements this feature without requiring
    to keep `S` copies of the mask of each dropout layer in a model. Instead, each time `reset_mask` is
    called, a dropout layer samples and stores a list of `S` random seeds. This list is cycled over
    indefinitely. At each one of the `S` forward passes, the `next(seed)` is called and dropout is applied
    with that specific seed. Note that this seed only affects the dropout mask generation via the `local_seed`
    context manager. Once the dropout mask is sampled, the random state of pytorch is reset again.

    This memory saving comes at the cost that the number of `S` of forward passes must match the number `S`
    of seeds so that forward pass 1 uses seed 1, etc. If they go out of sync, consistency is destroyed.
    This is not an issue when these dropout layers are accessed via an `Energizer` inference class since
    it takes care of keeping everything in sync. However, this issue must be considered when these dropout
    layers are accessed directly. In order to allow the user to change the number of forward passes
    interactively, the `reset_mask` method accepts an optional argument that allows to change the number
    (in our example, `S`) of masks that are sampled.

    Args:
        p (float): probability of an element to be zeroed. Default: 0.5
        inplace (bool): If set to True, will do this operation in-place.
        consistent (bool): If set to True, will use the consistent forward pass implementation.

    Keyword Args:
        num_inference_iters (int): Number of masks to sample.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False, consistent: bool = False, **consistent_kwargs):
        """Initialize a new dropout layer."""
        super().__init__(p=p, inplace=inplace)
        self.consistent = consistent
        self.num_inference_iters: Optional[int] = None
        self.seeds: Optional[cycle[int]] = None

        if self.consistent:
            if "num_inference_iters" not in consistent_kwargs:
                raise MisconfigurationException("When consistent is True, must pass 'num_inference_iters' as kwargs.")
            self.reset_mask(consistent_kwargs.get("num_inference_iters"))

    def reset_mask(self, num_inference_iters: Optional[int] = None) -> None:
        """Recreate the list of seeds for consistent sampling of dropout masks.

        Optionally it is possible to pass a new number of `num_inference_iters` so that the dropout layers
        can be reset to used this many number of seeds. This is crucial because consistency is preserved
        only when the number of forward passes matches the number of seeds. In cases in which the users
        interactively changes the former, this argument gives the possibility to change in the number of
        seeds interactively.

        Args:
            num_inference_iters (Optional[int]): If provided, changes the number of seeds. This must match
                the number of forward passes the user performs.
        """
        if num_inference_iters:
            self.num_inference_iters = num_inference_iters
        self.seeds = cycle(torch.randint(10, 10**8, size=(self.num_inference_iters,)).tolist())

    def eval(self) -> None:
        """Reset the mask when put in eval mode."""
        self.reset_mask()
        return super().eval()

    def _make_mask(self, x: Tensor) -> Tensor:
        """Create a mask (Tensor) of ones and it zeros out certain entries using this class dropout implementation."""
        shape = (self.num_inference_iters, *[x.shape[i] for i in range(x.ndim)])
        mask = torch.ones(shape, device=x.device, requires_grad=x.requires_grad)
        with local_seed(next(self.seeds)):  # type: ignore
            mask = self._forward(mask)
        return mask

    def _consistent_forward(self, x: Tensor) -> Tensor:
        self._mask = self._make_mask(x)
        return torch.mul(x, self._mask)

    def _forward(self, x: Tensor) -> Tensor:
        NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that is able to select the appropriate modality (consistent or not)."""
        if self.consistent and not self.training:
            return self._consistent_forward(x)
        return self._forward(x)


class Dropout(EnergizerDropoutLayer):
    """Patch of `torch.nn.Dropout`."""

    def _forward(self, input: Tensor) -> Tensor:
        """Calls the respective functional version with the `training` argument set to `True`."""
        return F.dropout(input=input, p=self.p, training=True, inplace=self.inplace)


class Dropout2d(EnergizerDropoutLayer):
    """Patch of `torch.nn.Dropout2d`."""

    def _forward(self, input: Tensor) -> Tensor:
        """Calls the respective functional version with the `training` argument set to `True`."""
        return F.dropout2d(input=input, p=self.p, training=True, inplace=self.inplace)


class Dropout3d(EnergizerDropoutLayer):
    """Patch of `torch.nn.Dropout3d`."""

    def _forward(self, input: Tensor) -> Tensor:
        """Calls the respective functional version with the `training` argument set to `True`."""
        return F.dropout3d(input=input, p=self.p, training=True, inplace=self.inplace)


class AlphaDropout(EnergizerDropoutLayer):
    """Patch of `torch.nn.AlphaDropout`."""

    def _forward(self, input: Tensor) -> Tensor:
        """Calls the respective functional version with the `training` argument set to `True`."""
        return F.alpha_dropout(input=input, p=self.p, training=True)


class FeatureAlphaDropout(EnergizerDropoutLayer):
    """Patch of `torch.nn.FeatureAlphaDropout`."""

    def _forward(self, input: Tensor) -> Tensor:
        """Calls the respective functional version with the `training` argument set to `True`."""
        return F.feature_alpha_dropout(input=input, p=self.p, training=True)


def patch_dropout_layers(
    module: nn.Module,
    prob: Optional[float] = None,
    inplace: bool = False,
    consistent: bool = False,
    **consistent_kwargs,
) -> nn.Module:
    """Replace dropout layers in a model with MCDropout layers.

    Args:
        module (nn.Module): The module in which dropout layers should be replaced.
        prob (float): If specified, this changes the dropout probability of all layers to `prob`. If `None` the dropout
            probability is the same as the original layer. Must be 0 < prob < 1.
        consistent (bool): If True, it uses the consistent version of dropout that fixes the mask across batches.
        inplace (bool): Whether to modify the module in place or return a copy of the module.

    Keyword Args:
        num_inference_iters (int): Number of masks to sample.

    Raises:
        MisconfigurationException if no layer is modified.

    Returns:
        A patched nn.Module which is either the same object passed in (if inplace = True) or a copy of that object.
    """
    if (prob is not None) and (prob < 0 or prob > 1):
        raise ValueError(f"Dropout probability must be 0 <= prob <= 1, not {prob}.")

    if not inplace:
        module = deepcopy(module)

    changed = _patch_dropout(module=module, prob=prob, consistent=consistent, **consistent_kwargs)
    if not changed:
        raise MisconfigurationException("The model should contain at least one dropout layer.")
    return module


def _patch_dropout(
    module: nn.Module, prob: Optional[float] = None, consistent: bool = False, **consistent_kwargs
) -> bool:
    """Recursively iterate over the children of a module and replace the dropout layers.

    Recursively replace dropout layers in a `torch.nn.Module` with the custom `Dropout` or `ConsistentDropout`
    modules needed to run MCDropout, i.e. to keep subsampling active even during evaluation.
    This function operates in-place.

    Args:
        module (nn.Module): The module to patch.
        patch_cls_dict: (Dict[str, _DropoutNd]): A dict with keys "dropout" and "dropout2d" whose values are the
            new classes to use as patches.
        prob (float): If specified, this changes the dropout probability of all layers.

    Returns:
        Flag indicating if at least a layer is modified.
    """
    changed = False
    for name, child in module.named_children():
        new_module: Optional[nn.Module] = None

        for dropout_layer in (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout):
            if isinstance(child, dropout_layer):

                # NOTE: `eval(dropout_layer.__name__)` works because the __name__ of the original dropout
                # layers is the same as the __name__ of the patched layers defined above
                new_module = eval(dropout_layer.__name__)(
                    p=prob if prob else child.p, inplace=child.inplace, consistent=consistent, **consistent_kwargs
                )

        if new_module is not None:
            changed = True
            module.add_module(name, new_module)

        # recursively apply to child
        changed |= _patch_dropout(module=child, prob=prob, consistent=consistent)
    return changed
