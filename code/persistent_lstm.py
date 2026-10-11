"""Persistent pure-LSTM baseline compatible with demo_rolling_GNN.py.

The manager intentionally mirrors the public rolling-model interface so the
rolling GNN script can switch between the multiscale model and this LSTM
baseline with minimal call-site changes. Unlike the multiscale manager, this
LSTM baseline trains each zone once and reuses the checkpoint for later rolling
hours.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ManagerConfig:
    hidden_size: int = 64
    num_layers: int = 1
    epochs_full: int = 50
    patience_full: int = 5
    lr_full: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 2.0
    sequence_length: int = 24 * 30
    forecast_length: int = 1
    p_drop: float = 0.2
    batch_size: int = 64
    context_hours: int = 800
    lambda_aux: float = 0.0
    K_mc_train: int = 5
    M_mc_test: int = 20
    min_delta: float = 1e-6


class PureLSTM(nn.Module):
    """Single-branch LSTM forecaster for hourly taxi demand."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        forecast_length: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, forecast_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last_hidden = self.dropout(hidden[-1])
        return self.head(last_hidden)

    def mc_predict(self, x: torch.Tensor, samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(max(1, int(samples))):
                preds.append(self.forward(x).unsqueeze(0))
        pred_stack = torch.cat(preds, dim=0)
        return pred_stack.mean(dim=0), pred_stack.std(dim=0, unbiased=False)


class PureLSTMModelManager:
    """Per-zone persistent LSTM manager using train-once/reuse behavior."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints_lstm",
        cfg: Optional[ManagerConfig] = None,
        hidden_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        cfg = cfg or ManagerConfig()
        if hidden_size is not None:
            cfg = replace(cfg, hidden_size=hidden_size)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def _forecast_delta(self) -> pd.Timedelta:
        return pd.Timedelta(hours=self.cfg.forecast_length)

    def _model_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"lstm_zone_{int(zone_id)}.pt"

    def _scaler_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"lstm_scaler_zone_{int(zone_id)}.pkl"

    def _meta_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"lstm_meta_zone_{int(zone_id)}.json"

    def has_checkpoint(self, zone_id: int) -> bool:
        return (
            self._model_path(zone_id).exists()
            and self._scaler_path(zone_id).exists()
            and self._meta_path(zone_id).exists()
        )

    def _new_model(self) -> PureLSTM:
        return PureLSTM(
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.p_drop,
            forecast_length=self.cfg.forecast_length,
        ).to(self.device)

    def _save(self, zone_id: int, model: nn.Module, scaler: MinMaxScaler) -> None:
        torch.save(model.state_dict(), self._model_path(zone_id))
        with open(self._scaler_path(zone_id), "wb") as fh:
            pickle.dump(scaler, fh)

    def _load(self, zone_id: int) -> Tuple[PureLSTM, MinMaxScaler]:
        model = self._new_model()
        state = torch.load(self._model_path(zone_id), map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        with open(self._scaler_path(zone_id), "rb") as fh:
            scaler = pickle.load(fh)
        return model, scaler

    def _save_meta(self, zone_id: int, context_end: pd.Timestamp) -> None:
        meta = {"last_trained_context_end": str(context_end)}
        with open(self._meta_path(zone_id), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)

    def _load_meta(self, zone_id: int) -> Optional[pd.Timestamp]:
        path = self._meta_path(zone_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        ts = meta.get("last_trained_context_end") or meta.get("last_trained_until")
        return pd.Timestamp(ts) if ts is not None else None

    @staticmethod
    def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
        if "datetime" in df.columns:
            return df
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["pickup_datetime"]).dt.floor("h")
        return df

    def _prepare_zone_series(
        self,
        df: pd.DataFrame,
        zone_id: int,
        end_inclusive: pd.Timestamp,
    ) -> pd.DataFrame:
        df = self._ensure_datetime(df)
        context_hours = max(
            self.cfg.context_hours,
            self.cfg.sequence_length + self.cfg.forecast_length + 1,
        )
        start_date = end_inclusive - pd.Timedelta(hours=context_hours)
        mask = (
            (df["PULocationID"] == zone_id)
            & (df["datetime"] >= start_date)
            & (df["datetime"] <= end_inclusive)
        )
        hourly_counts = df.loc[mask].groupby("datetime").size()
        hourly_index = pd.date_range(start=start_date, end=end_inclusive, freq="h")
        hourly = (
            hourly_counts.reindex(hourly_index, fill_value=0)
            .rename("passenger_count")
            .rename_axis("datetime")
            .reset_index()
        )
        hourly["passenger_count"] = hourly["passenger_count"].astype(float)
        return hourly

    @staticmethod
    def _fit_scaler_hist(hourly: pd.DataFrame, fit_until_exclusive: pd.Timestamp) -> MinMaxScaler:
        scaler = MinMaxScaler()
        hist = hourly[hourly["datetime"] < fit_until_exclusive]
        values = hist[["passenger_count"]].astype(float)

        if values.empty:
            scaler.fit(pd.DataFrame({"passenger_count": [0.0, 1.0]}))
            return scaler

        vmin = float(values["passenger_count"].min())
        vmax = float(values["passenger_count"].max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            base = 0.0 if not np.isfinite(vmin) else vmin
            scaler.fit(pd.DataFrame({"passenger_count": [base, base + 1.0]}))
            return scaler

        scaler.fit(values)
        return scaler

    def _build_training_arrays(
        self,
        hourly: pd.DataFrame,
        scaler: MinMaxScaler,
    ) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = self.cfg.sequence_length
        forecast_len = self.cfg.forecast_length
        hourly = hourly.copy()
        hourly["passenger_count_scaled"] = scaler.transform(hourly[["passenger_count"]])
        series = hourly["passenger_count_scaled"]

        x_values, y_values = [], []
        for start_idx in range(len(hourly) + 1 - seq_len - forecast_len):
            target_start = start_idx + seq_len
            target_end = target_start + forecast_len
            x_values.append(series.iloc[start_idx:target_start].values)
            y_values.append(series.iloc[target_start:target_end].values)

        if not x_values:
            raise ValueError("Insufficient history to build LSTM training windows.")

        X = np.asarray(x_values, dtype=np.float32)[..., None]
        y = np.asarray(y_values, dtype=np.float32)
        return X, y

    def _build_inference_window(
        self,
        hourly: pd.DataFrame,
        scaler: MinMaxScaler,
        context_end: pd.Timestamp,
    ) -> torch.Tensor:
        seq_len = self.cfg.sequence_length
        hourly = hourly.copy()
        hourly["passenger_count_scaled"] = scaler.transform(hourly[["passenger_count"]])
        idx_map = {ts: idx for idx, ts in enumerate(hourly["datetime"])}
        if context_end not in idx_map:
            raise ValueError("context_end missing from hourly series.")

        end_idx = idx_map[context_end]
        start_idx = end_idx - seq_len + 1
        if start_idx < 0:
            raise ValueError("Not enough history to assemble LSTM inference window.")

        values = hourly["passenger_count_scaled"].iloc[start_idx : end_idx + 1].values
        return torch.tensor(values, dtype=torch.float32, device=self.device).view(1, -1, 1)

    def _train_arrays(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        criterion: nn.Module,
        use_patience: bool,
    ) -> None:
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(
            dataset,
            batch_size=min(self.cfg.batch_size, max(1, len(dataset))),
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=self.cfg.weight_decay,
        )
        best_loss = float("inf")
        best_state = None
        patience_ctr = 0

        for _ in range(max(1, int(epochs))):
            model.train()
            total_loss = 0.0
            sample_count = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                optimizer.step()
                total_loss += float(loss.item()) * xb.size(0)
                sample_count += xb.size(0)

            epoch_loss = total_loss / max(1, sample_count)
            if not use_patience:
                continue

            if epoch_loss < best_loss - self.cfg.min_delta:
                best_loss = epoch_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.patience_full:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

    @staticmethod
    def _inverse_values(values_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        original_shape = values_scaled.shape
        values = np.asarray(values_scaled, dtype=np.float32).reshape(-1, 1)
        return scaler.inverse_transform(values).reshape(original_shape)

    @staticmethod
    def _inverse_std_minmax(std_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        data_range = float(scaler.data_max_[0] - scaler.data_min_[0])
        return std_scaled * data_range

    def train_once(self, df: pd.DataFrame, zone_id: int, context_end: pd.Timestamp) -> None:
        if self.has_checkpoint(zone_id):
            return

        hourly = self._prepare_zone_series(df, zone_id, context_end)
        scaler = self._fit_scaler_hist(
            hourly,
            fit_until_exclusive=context_end + pd.Timedelta(hours=1),
        )
        X, y = self._build_training_arrays(hourly, scaler)
        model = self._new_model()
        self._train_arrays(
            model=model,
            X=X,
            y=y,
            epochs=self.cfg.epochs_full,
            lr=self.cfg.lr_full,
            criterion=nn.MSELoss(),
            use_patience=True,
        )
        self._save(zone_id, model, scaler)
        self._save_meta(zone_id, context_end)

    def predict(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> float:
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(f"Zone {zone_id} has no LSTM checkpoint; train first.")

        model, scaler = self._load(zone_id)
        context_end = target_date - self._forecast_delta
        hourly = self._prepare_zone_series(df, zone_id, context_end)
        X_last = self._build_inference_window(hourly, scaler, context_end)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_last).cpu().numpy()
        pred = self._inverse_values(pred_scaled, scaler)[0, 0]
        return float(max(0.0, pred))

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        zone_id: int,
        target_date: pd.Timestamp,
    ) -> Tuple[float, float, Dict[str, float]]:
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(f"Zone {zone_id} has no LSTM checkpoint; train first.")

        model, scaler = self._load(zone_id)
        context_end = target_date - self._forecast_delta
        hourly = self._prepare_zone_series(df, zone_id, context_end)
        X_last = self._build_inference_window(hourly, scaler, context_end)

        mean_scaled, std_scaled = model.mc_predict(X_last, self.cfg.M_mc_test)
        mean_np = mean_scaled.cpu().numpy()
        std_np = std_scaled.cpu().numpy()
        point = self._inverse_values(mean_np, scaler)[0, 0]
        std_orig = self._inverse_std_minmax(std_np, scaler)[0, 0]
        diagnostics = {"lstm_mc_variance": float(std_orig**2)}
        return float(max(0.0, point)), float(max(0.0, std_orig)), diagnostics

    def train_and_predict_if_needed(
        self,
        df: pd.DataFrame,
        zone_id: int,
        target_date: pd.Timestamp,
        auto_train: bool = True,
    ) -> float:
        context_end = target_date - self._forecast_delta
        if not self.has_checkpoint(zone_id):
            if not auto_train:
                raise FileNotFoundError(f"No LSTM checkpoint for zone {zone_id}.")
            self.train_once(df, zone_id, context_end)

        return self.predict(df, zone_id, target_date)


MultiScaleModelManager = PureLSTMModelManager
