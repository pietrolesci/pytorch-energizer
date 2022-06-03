from mnist_example import MNISTModel, MNISTDataModule
from energizer.learners.base import Deterministic
from energizer.trainer import Trainer
from torch import Tensor
from pytorch_lightning import LightningModule
from energizer.learners.acquisition_functions import entropy


class Learner(Deterministic):
    def __init__(self, learner: LightningModule, query_size: int):
        super().__init__(learner, query_size)

    def pool_step(self, batch, *args, **kwargs) -> Tensor:
        x, _ = batch
        logits = self(x)
        scores = entropy(logits)
        return scores


if __name__ == "__main__":
    datamodule = MNISTDataModule(batch_size=32)
    datamodule.setup()

    model = MNISTModel()
    active_learner = Learner(model, query_size=1)

    trainer = Trainer(max_epochs=1)
    trainer.active_fit(active_learner, datamodule=datamodule)
