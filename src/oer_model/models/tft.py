"""Temporal Fusion Transformer wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ForecastModel

try:  # pragma: no cover - optional dependencies
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    HAS_TFT = True
except ImportError:  # pragma: no cover
    HAS_TFT = False


@dataclass
class TFTParams:
    """Hyperparameters for the Temporal Fusion Transformer."""

    max_encoder_length: int = 18
    max_prediction_length: int = 6
    hidden_size: int = 32
    attention_head_size: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    accelerator: str = "cpu"
    devices: int | List[int] | None = None


class TemporalFusionTransformerModel(ForecastModel):
    """Convenience wrapper around pytorch-forecasting's TFT."""

    def __init__(self, name: str = "tft", params: Optional[TFTParams | Dict[str, Any]] = None, known_future_covariates: Optional[List[str]] = None):
        if not HAS_TFT:
            raise ImportError("pytorch-forecasting and pytorch-lightning are required for TFT")
        super().__init__(name)
        if params is None:
            self.params = TFTParams()
        elif isinstance(params, dict):
            self.params = TFTParams(**params)
        else:
            self.params = params
        self.known_future_covariates = known_future_covariates or []
        self._trainer: Optional[pl.Trainer] = None
        self._model: Optional[TemporalFusionTransformer] = None
        self._dataset: Optional[TimeSeriesDataSet] = None
        self._target_name: Optional[str] = None

    def _prepare_dataset(self, X: pd.DataFrame, y: pd.Series) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        data = X.copy()
        self._target_name = y.name or "target"
        data[self._target_name] = y
        data = data.sort_index()
        data.index.name = "date"
        data = data.reset_index()
        data["time_idx"] = np.arange(len(data))
        data["series_id"] = "oer"

        max_prediction_length = self.params.max_prediction_length
        cutoff = data["time_idx"].max() - max_prediction_length * 2
        cutoff = max(cutoff, self.params.max_encoder_length)

        covariates = list(X.columns)
        encoder_length = self.params.max_encoder_length

        training = TimeSeriesDataSet(
            data[data["time_idx"] <= cutoff],
            time_idx="time_idx",
            target=self._target_name,
            group_ids=["series_id"],
            max_encoder_length=encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=[*covariates, *self.known_future_covariates],
            time_varying_unknown_reals=[self._target_name],
            target_normalizer=GroupNormalizer(groups=["series_id"]),
            add_relative_time_idx=True,
            add_target_scales=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            data[data["time_idx"] > cutoff],
            stop_randomization=True,
        )
        return training, validation

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TemporalFusionTransformerModel":
        training, validation = self._prepare_dataset(X, y)
        train_loader = training.to_dataloader(train=True, batch_size=self.params.batch_size)
        val_loader = validation.to_dataloader(train=False, batch_size=self.params.batch_size)

        self._dataset = training
        self._model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.params.learning_rate,
            hidden_size=self.params.hidden_size,
            attention_head_size=self.params.attention_head_size,
            dropout=self.params.dropout,
            loss=None,
        )
        accelerator = self.params.accelerator
        devices = self.params.devices or (1 if accelerator == "gpu" else None)
        self._trainer = pl.Trainer(
            max_epochs=self.params.max_epochs,
            gradient_clip_val=self.params.gradient_clip_val,
            accelerator=accelerator,
            devices=devices,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
        )
        self._trainer.fit(self._model, train_loader, val_loader)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._model or not self._dataset or not self._target_name:
            raise RuntimeError("Model must be fitted before predicting")
        data = X.copy().sort_index()
        data.index.name = "date"
        data = data.reset_index()
        data["time_idx"] = np.arange(len(data))
        data["series_id"] = "oer"
        dataset = TimeSeriesDataSet.from_dataset(
            self._dataset,
            data,
            stop_randomization=True,
        )
        loader = dataset.to_dataloader(train=False, batch_size=self.params.batch_size)
        preds = self._trainer.predict(self._model, loader)
        return np.concatenate([p.numpy().ravel() for p in preds])

    def get_params(self) -> Dict[str, Any]:
        return self.params.__dict__
