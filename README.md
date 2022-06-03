# Pytorch-Energizer


[![pypi](https://img.shields.io/pypi/v/pytorch-energizer.svg)](https://pypi.org/project/pytorch-energizer/)
[![python](https://img.shields.io/pypi/pyversions/pytorch-energizer.svg)](https://pypi.org/project/pytorch-energizer/)
[![Build Status](https://github.com/pietrolesci/pytorch-energizer/actions/workflows/dev.yml/badge.svg)](https://github.com/pietrolesci/pytorch-energizer/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/pietrolesci/pytorch-energizer/branch/main/graphs/badge.svg)](https://codecov.io/github/pietrolesci/pytorch-energizer)



An active learning library for Pytorch


* Documentation: <https://pietrolesci.github.io/pytorch-energizer>
* GitHub: <https://github.com/pietrolesci/pytorch-energizer>
* PyPI: <https://pypi.org/project/pytorch-energizer/>
* Free software: MIT


## Features

`Energizer` allows training Pytorch models using active learning. Being based on Pytorch-Lightning, it can easily scale to multi-node/multi-gpu settings. Also, importantly, abiding to the light-weight Pytorch-Lightning API allows the this library to have a tidy interface and completely avoid boilerplate training code.

The core principle underlying Energizer is composability. Everything in the library revolves around the `EnergizerStrategy` which puts together a `base_learner` (the model we want to actively train), an `inference_module` (how the model should behave when run on the pool dataset), and the active learning loop hyper-parameters.

 For example, assume you have the following model

```python
class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=4)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        return self.backbone(**batch).logits

    def step(self, batch, *args, **kwargs):
        y = batch.pop("labels")
        y_hat = self(batch)
        return self.loss(y_hat, y)

    def training_step(self, batch, *args, **kwargs):
        loss = self.step(batch, *args, **kwargs)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss = self.step(batch, *args, **kwargs)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, *args, **kwargs):
        loss = self.step(batch, *args, **kwargs)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


base_learner = Model()
```

The first step to let `Energizer` know how this model should behave at inference time on the pool dataset. This is easily done by wrapping it into an `EnergizerInference` module. Let's say that you want to use MC-Dropout. You can then do

```python
from energizer.inference import MCDropout

inference_module = MCDropout(
    num_inference_iters: int = 10,
    consistent: bool = False,
    prob: Optional[float] = 0.1,
    inplace: bool = True,
)

# this will patch all Dropout layers
inference_module.connect(base_learner)  # NOTE: when used inside an `EnergizerStrategy`
                                        # this will be done automatically
```

Now whenever `inference_module(x)` is called it will perform `num_inference_iter` forward passes with the dropout layers activated and collect the resulting list of logits, as prescribed by the MC-Dropout technique. To actually tell how to score instances from the pool and how to select indices, we can wrap the inference module into an `EnergyStrategy`. For this example, let's assume you want to use the entropy strategy that selects the instances with the highest entropy of the logits

```python
from energizer.strategies import EntropyStrategy

al_strategy = EntropyStrategy(inference_module=inference_module)
```

Under the hood this will call the `inference_module.forward()`. Since our inference module performs MC-Dropout, when will automatically use the _expected_ entropy.
Each `EnergizerStrategy` is a `LightingModule` whose `test_step` has been tailored to perform scoring and selection of the instances to label. In practice, each batch from the pool dataset is scored. The top-k scores are kept in memory at each iteration alongside their indices. This avoids scoring the entire pool first and then computing the top-k, which can be unfeasible when the pool is very big and does not play nicely with distributed settings.

The missing piece is the actual active learning loop definition. In Energizer this is handled by the `ActiveLearningLoop`, which is a subclass of the Lighting `FitLoop`. It can be defined as shown below. Also, the next step shows how to do everything (define an inference module, strategy, and loop) in one go.

```python
from energizer.loops import ActiveLearningLoop
from energizer.strategies import EntropyStrategy
from pytorch_lightning import Trainer


# define model
base_learner = Model()

# define active learning loop, strategy, and inference module (no need to call `.connect()`)
active_learning_loop = ActiveLearningLoop(
    al_strategy=EntropyStrategy(
        inference_module=MCDropout(
            num_inference_iters: int = 10,
            consistent: bool = False,
            prob: Optional[float] = 0.1,
            inplace: bool = True,
        )
    ),
    query_size: int = 2,             # number of instance to label at each round
    reset_weights: bool = True,      # should we reset the model weights after each iteration?
    n_epochs_between_labelling: int = 3,  # how many training epochs on the labelled data
)

trainer = Trainer(max_epochs=10)

# Connect to the default fit_loop of the trainer to extract info, e.g. max_epochs
# NOTE: there is no need to call `.connect()` on the strategy or on the inference module,
# everything is handled by this `.connect()` call
active_learning_loop.connect(trainer)

# replace the original fit_loop with the active_learning_loop
trainer.fit_loop = active_learning_loop

# fit model with active learning
trainer.fit(base_learner, datamodule=dm)
```



## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
