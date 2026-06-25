"""Shared evaluator interface for bike-demand submissions."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract interface used by the local and official evaluators.

    A valid submission folder contains a fixed wrapper in ``predict.py``:

        class Model(BaseModel):
            def load(self, weights_path: str) -> None: ...
            def predict(self, test_df: pd.DataFrame) -> np.ndarray: ...

    The evaluator constructs that wrapper and calls:

        model = Model()
        model.load("submissions/<team_name>/weights.joblib")
        predictions = model.predict(test_targets_df)

    ``test_targets_df`` is station-hour target data, not ride-level data. Each
    row asks for the predicted number of rides starting from one station during
    one hour. It must not contain the true ``demand`` label.
    """

    @abstractmethod
    def load(self, weights_path: str) -> None:
        """Load all trained artifacts needed for prediction."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return one numeric prediction per row of ``test_df``.

        Implementations should not mutate ``test_df`` and should not require a
        ``demand`` column. Predictions should be finite and non-negative; the
        evaluator also clips negative values to zero before scoring.
        """
        raise NotImplementedError
