"""Persistence utilities for orchestrator runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import logging
import pandas as pd

from .evaluation import EvaluationRecord

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency
    import pyarrow  # noqa: F401

    PARQUET_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PARQUET_AVAILABLE = False


@dataclass
class RunLogger:
    output_dir: Path | str
    jsonl_name: str = 'evaluations.jsonl'
    parquet_name: str = 'evaluations.parquet'
    enable_parquet: bool = True

    def __post_init__(self) -> None:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_path
        self.jsonl_path = output_path / self.jsonl_name
        self.parquet_path = output_path / self.parquet_name
        self._parquet_buffer: list[dict[str, object]] = []

    def log(self, record: EvaluationRecord) -> None:
        with self.jsonl_path.open('a') as handle:
            handle.write(json.dumps(record.jsonable()) + '\n')
        if self.enable_parquet and PARQUET_AVAILABLE:
            self._parquet_buffer.append(record.parquet_row())
        elif self.enable_parquet and not PARQUET_AVAILABLE:
            logger.debug('Parquet logging requested but pyarrow is not installed')

    def extend(self, records: Iterable[EvaluationRecord]) -> None:
        for record in records:
            self.log(record)

    def flush(self) -> None:
        if not self._parquet_buffer:
            return
        if not PARQUET_AVAILABLE:
            self._parquet_buffer.clear()
            return
        df = pd.DataFrame(self._parquet_buffer)
        if self.parquet_path.exists():
            existing = pd.read_parquet(self.parquet_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(self.parquet_path, index=False)
        self._parquet_buffer.clear()


__all__ = ['RunLogger']
