"""
Time-series Dense Encoder (TSMixer)
------
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from darts.logging import get_logger
from darts.models.forecasting.pl_forecasting_module import PLMixedCovariatesModule
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


logger = get_logger(__name__)


class _ReversibleInstanceNorm(nn.Module):
    def __init__(self, axis, input_dim, eps=1e-5, affine=True):
        super().__init__()
        self.axis = axis
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, x, mode, target_slice=None):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=self.axis, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=self.axis, keepdim=True) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = (x.transpose(2, 1) * self.affine_weight).transpose(2, 1)
            x = (x.transpose(2, 1) + self.affine_bias).transpose(2, 1)
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = (x.transpose(2, 1) - self.affine_bias[target_slice]).transpose(2, 1)
            x = (x.transpose(2, 1) / self.affine_weight[target_slice]).transpose(2, 1)
        x = x * self.stdev
        x = x + self.mean
        return x


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_chunk_length: int,
        dropout: float,
        hidden_size: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.norm = nn.BatchNorm1d(self.input_dim)

        self.temporal_linear = nn.Sequential(
            nn.Linear(input_chunk_length, input_chunk_length),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.feature_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.input_dim),
            nn.Dropout(self.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch_size, input_chunk_length, input_dim)

        # temporal_norm = self.norm(x.transpose(1, 2))
        # x = x + self.temporal_linear(temporal_norm).transpose(1, 2)
        x = x + self.temporal_linear(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.feature_linear(x)

        return x


class _TSMixerModel(PLMixedCovariatesModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        num_mixer_layers: int,
        hidden_size: int,
        dropout: float,
        **kwargs,
    ):
        """Pytorch module implementing the TSMixer architecture.

        Parameters
        ----------
        input_dim
            The number of input components (target + optional past covariates + optional future covariates).
        output_dim
            Number of output components in the target.
        future_cov_dim
            Number of future covariates.
        static_cov_dim
            Number of static covariates.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        num_encoder_layers
            Number of stacked Residual Blocks in the encoder.
        num_decoder_layers
            Number of stacked Residual Blocks in the decoder.
        decoder_output_dim
            The number of output components of the decoder.
        hidden_size
            The width of the hidden layers in the encoder/decoder Residual Blocks.
        temporal_decoder_hidden
            The width of the hidden layers in the temporal decoder.
        temporal_width
            The width of the future covariate embedding space.
        dropout
            Dropout probability
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x
            Tuple of Tensors `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and
            `x_future`is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`
        Outputs
        -------
        y
            Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`

        """

        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.num_mixer_layers = num_mixer_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.rev_in_norm = _ReversibleInstanceNorm(
            axis=-2,
            input_dim=self.input_chunk_length,
        )

        self.mixer_stack = nn.Sequential(
            *[
                _ResidualBlock(
                    input_dim=self.input_dim,
                    input_chunk_length=self.input_chunk_length,
                    dropout=self.dropout,
                    hidden_size=self.hidden_size,
                )
                for _ in range(self.num_mixer_layers)
            ],
        )

        self.temporal_projection = nn.Linear(
            self.input_chunk_length,
            self.output_chunk_length * self.output_dim * self.nr_params,
        )

    def forward(
        self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """TSMixer model forward pass.
        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`
        Returns
        -------
        torch.Tensor
            The output Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`
        """

        # x has shape (batch_size, input_chunk_length, input_dim)
        # x_future_covariates has shape (batch_size, input_chunk_length, future_cov_dim)
        # x_static_covariates has shape (batch_size, static_cov_dim)
        x, x_future_covariates, x_static_covariates = x_in

        x = self.rev_in_norm(x, mode="norm")

        y_hat = self.mixer_stack(x)

        y_hat = self.temporal_projection(y_hat.transpose(1, 2)).transpose(1, 2)

        y_hat = self.rev_in_norm(y_hat, mode="denorm")

        y_hat = y_hat.view(
            -1, self.output_chunk_length, self.output_dim, self.nr_params
        )

        return y_hat


class TSMixerModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        hidden_size: int = 128,
        num_mixer_layers: int = 2,
        dropout: float = 0.1,
        use_static_covariates: bool = True,
        **kwargs,
    ):
        """An implementation of the TSMixer model, as presented in [1]_.

        TSMixer is similar to Transformers (implemented in :class:`TransformerModel`),
        but attempts to provide better performance at lower computational cost by introducing
        multilayer perceptron (MLP)-based encoder-decoders without attention.

        The model is implemented as a :class:`MixedCovariatesTorchModel`, which means that it supports
        both past and future covariates, as well as static covariates. At this time the model does not support
        probabilistic forecasts. The original paper does not consider the use of past covariates, so we assume
        that they are passed to the encoder as-is.

        The encoder and decoder are implemented as a series of residual blocks. The number of residual blocks in
        the encoder and decoder can be controlled via ``num_encoder_layers`` and ``num_decoder_layers`` respectively.
        The width of the layers in the residual blocks can be controlled via ``hidden_size``. Similarly, the width
        of the layers in the temporal decoder can be controlled via ``temporal_decoder_hidden``.

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the forecast of the model.
        num_encoder_layers
            The number of residual blocks in the encoder.
        num_decoder_layers
            The number of residual blocks in the decoder.
        decoder_output_dim
            The dimensionality of the output of the decoder.
        hidden_size
            The width of the layers in the residual blocks of the encoder and decoder.
        temporal_width
            The width of the layers in the future covariate projection residual block.
        temporal_decoder_hidden
            The width of the layers in the temporal decoder.
        dropout
            The dropout probability to be used in fully connected layers. This is compatible with Monte Carlo dropout
            at inference time for model uncertainty estimation (enabled with ``mc_dropout=True`` at
            prediction time).
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        likelihood
            One of Darts' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for
            probabilistic forecasts. Default: ``None``.
        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
            for specifying a learning rate). Otherwise, the default values of the selected ``optimizer_cls``
            will be used. Default: ``None``.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds
            to using a constant learning rate. Default: ``None``.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
        batch_size
            Number of time series (input and output sequences) used in each training pass. Default: ``32``.
        n_epochs
            Number of epochs over which to train the model. Default: ``100``.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09_53_32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            Default: current working directory.
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.
        force_reset
            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded). Default: ``False``.
        save_checkpoints
            Whether or not to automatically save the untrained model and checkpoints from training.
            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
            :func:`save()` and loaded using :func:`load()`. Default: ``False``.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.
            Default: ``None``.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the
            supported kwargs. Default: ``None``.
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",
            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:


            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.

            For more info, see here:
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus

            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_

            .. highlight:: python
            .. code-block:: python

                from pytorch_lightning.callbacks.early_stopping import EarlyStopping

                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
                # a period of 5 epochs (`patience`)
                my_stopper = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=0.05,
                    mode='min',
                )

                pl_trainer_kwargs={"callbacks": [my_stopper]}
            ..

            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional
            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.
        show_warnings
            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.

        References
        ----------
        .. [1] A. Das et al. "Long-term Forecasting with TSMixer: Time-series Dense Encoder",
               http://arxiv.org/abs/2304.08424
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self.hidden_size = hidden_size
        self.num_mixer_layers = num_mixer_layers

        self._considers_static_covariates = use_static_covariates

        self.dropout = dropout

    @property
    def supports_static_covariates(self) -> bool:
        return True

    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> torch.nn.Module:
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        # target, past covariates, historic future covariates
        input_dim = (
            past_target.shape[1]
            + (past_covariates.shape[1] if past_covariates is not None else 0)
            + (
                historic_future_covariates.shape[1]
                if historic_future_covariates is not None
                else 0
            )
        )

        output_dim = future_target.shape[1]

        future_cov_dim = (
            future_covariates.shape[1] if future_covariates is not None else 0
        )
        static_cov_dim = (
            static_covariates.shape[0] * static_covariates.shape[1]
            if static_covariates is not None
            else 0
        )

        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _TSMixerModel(
            input_dim=input_dim,
            output_dim=output_dim,
            future_cov_dim=future_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            hidden_size=self.hidden_size,
            num_mixer_layers=self.num_mixer_layers,
            dropout=self.dropout,
            **self.pl_module_params,
        )
