from typing import Any, Dict, List, Optional, Union

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset, Subset


class DataloaderToDataModule(LightningDataModule):
    """Given individual dataloaders, create a datamodule."""

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloaders: Union[DataLoader, List[DataLoader]],
        test_dataloaders: Union[DataLoader, List[DataLoader]],
    ) -> None:
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloaders = val_dataloaders
        self._test_dataloaders = test_dataloaders

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._val_dataloaders

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._test_dataloaders


class ActiveDataModule(LightningDataModule):
    train_fold: Optional[Dataset] = None
    pool_fold: Optional[Dataset] = None

    def __init__(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        if train_dataloader is None and datamodule is None:
            raise MisconfigurationException("Either `train_dataloader` or `datamodule` argument should be provided")
        if train_dataloader is not None and datamodule is not None:
            raise MisconfigurationException(
                "Only one of `train_dataloader` and `datamodule` argument should be provided"
            )
        if train_dataloader is not None:
            self.datamodule = DataloaderToDataModule(train_dataloader, val_dataloaders, test_dataloaders)
        if datamodule is not None:
            self.datamodule = datamodule

        super().__init__()

        self._train_dataloader_stats = None
        self._eval_dataloader_stats = None
        self.setup_folds()

    def get_dataloader_stats(self, dataloader: DataLoader) -> Optional[Dict[str, Any]]:
        # TODO: look at how lightning does this inspection of the parameters, might work better
        if dataloader is not None:
            dataloader = dataloader[0] if isinstance(dataloader, List) else dataloader
            return {
                "batch_size": dataloader.batch_size,
                "num_workers": dataloader.num_workers,
                "collate_fn": dataloader.collate_fn,
                "pin_memory": dataloader.pin_memory,
                "drop_last": dataloader.drop_last,
                "timeout": dataloader.timeout,
                "worker_init_fn": dataloader.worker_init_fn,
                "prefetch_factor": dataloader.prefetch_factor,
                "persistent_workers": dataloader.persistent_workers,
            }

    @property
    def labelled_size(self) -> int:
        return len(self.train_fold)

    @property
    def pool_size(self) -> int:
        return len(self.pool_fold)

    @property
    def train_dataloader_stats(self) -> Dict[str, Any]:
        return self._train_dataloader_stats

    @property
    def eval_dataloader_stats(self) -> Dict[str, Any]:
        return self._eval_dataloader_stats

    @property
    def has_labelled_data(self) -> bool:
        """Checks whether there are labelled data available."""
        return len(self.train_fold) > 0

    @property
    def has_unlabelled_data(self) -> bool:
        """Checks whether there are data to be labelled."""
        return len(self.pool_fold) > 0

    @property
    def last_labelling_step(self) -> int:
        """Returns the number of the last active learning step."""
        return int(self.train_mask.max())

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.datamodule.setup(stage)

    def setup_folds(self) -> None:
        train_dl = self.datamodule.train_dataloader()
        train_dataset = train_dl.dataset
        self._train_dataloader_stats = self.get_dataloader_stats(train_dl)
        self.train_fold = Subset(train_dataset, [])
        self.pool_fold = Subset(train_dataset, [])

        eval_dl = self.datamodule.test_dataloader() or self.datamodule.val_dataloader()
        self._eval_dataloader_stats = self.get_dataloader_stats(eval_dl)
        self._eval_dataloader_stats = self._eval_dataloader_stats or self._train_dataloader_stats

        # states
        self.train_mask = np.zeros((len(train_dataset),), dtype=int)
        self.setup_fold_index()

    def label(self, pool_idx: Union[int, List[int]]) -> None:
        """Moves instances at index `pool_idx` from the `train_fold` to the `pool_fold`.

        Args:
            pool_idx (List[int]): The index (relative to the pool_fold, not the overall data) to label.
        """
        assert isinstance(pool_idx, int) or isinstance(pool_idx, list), ValueError(
            f"`pool_idx` must be of type `int` or `List[int]`, not {type(pool_idx)}."
        )

        pool_idx = [pool_idx] if isinstance(pool_idx, int) else pool_idx

        if len(np.unique(pool_idx)) > len(self.pool_fold):
            raise ValueError(
                f"Pool has {len(self.pool_fold)} instances, cannot label {len(np.unique(pool_idx))} instances."
            )
        if max(pool_idx) > len(self.pool_fold):
            raise ValueError(f"Cannot label instance {max(pool_idx)} for pool dataset of size {len(self.pool_fold)}.")

        # acquire labels
        indices = self.pool_to_original(pool_idx)
        self.train_mask[indices] = self.last_labelling_step + 1  # current labelling step
        self.setup_fold_index()

    def pool_to_original(self, pool_idx: List[int]) -> List[int]:
        """Transform indices in `pool_fold` to indices in the original `dataset`."""
        return [self.pool_fold.indices[i] for i in np.unique(pool_idx)]

    def setup_fold_index(self) -> None:
        """Updates indices in each `torch.utils.data.Subset`.

        It stores the indices as `List[int]` and updates the relative `{labelled, pool}_size` attributes.
        """
        self.pool_fold.indices = np.where(self.train_mask == 0)[0].tolist()
        self.train_fold.indices = np.where(self.train_mask > 0)[0].tolist()

    def reset_at_labelling_step(self, labelling_step: int) -> None:
        """Resets the dataset at the state in which it was after the last `labelling_step`.

        To bring it back to the latest labelling step, simply run
        `obj.reset_at_labellling_step(obj.last_labelling_step)`.

        Args:
            labelling_step (int): The labelling step at which the dataset status should be brought.
        """
        if labelling_step > self.last_labelling_step:
            raise ValueError(
                f"`labelling_step` ({labelling_step}) must be less than the total number "
                f"of labelling steps performed ({self.last_labelling_step})."
            )
        self.pool_pool.indices = np.where((self.train_mask == 0) | (self.train_mask > labelling_step))[0].tolist()
        self.train_pool.indices = np.where((self.train_mask > 0) & (self.train_mask <= labelling_step))[0].tolist()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, **self.train_dataloader_stats)

    def val_dataloader(self) -> Optional[DataLoader]:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> Optional[DataLoader]:
        return self.datamodule.test_dataloader()

    def pool_dataloader(self) -> DataLoader:
        return DataLoader(self.pool_fold, **self.eval_dataloader_stats)
