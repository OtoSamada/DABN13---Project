from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass(slots = True)
class SearchResult:
    algorithm: str
    best_params: Dict[str, Any]
    best_score: float
    scoring: str
    cv: int


class ModelSearch:
    """
    Generic hyperparameter search supporting any sklearn-compatible estimator.
    """

    def __init__(
        self,
        estimator,
        param_grid: Dict[str, Sequence[Any]],
        numeric_features: List[str],
        categorical_features: List[str],
        scale_numeric: bool = True,
        scoring: str = "f1",
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
    ) -> None:
        
        self.estimator = estimator
        self.param_grid_raw = param_grid
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scale_numeric = scale_numeric
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

        numeric_steps = []
        if scale_numeric and numeric_features:
            numeric_steps.append(("scaler", StandardScaler()))

        transformers = []
        if numeric_features:
            transformers.append(("num", Pipeline(numeric_steps), numeric_features))
        if categorical_features:
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown = "ignore", sparse_output = True),
                categorical_features
            ))

        self.preprocessor = ColumnTransformer(transformers, remainder = "drop", sparse_threshold = 0.3)

        self.pipeline = Pipeline(
            steps = [
                ("prep", self.preprocessor),
                ("est", self.estimator),
            ]
        )

        # Map param grid to est__ namespace
        self.param_grid = {f"est__{k}": v for k, v in param_grid.items()}
        self._gs: Optional[GridSearchCV] = None

    def fit(self, X: pd.DataFrame, y) -> SearchResult:
        self._gs = GridSearchCV(
            estimator = self.pipeline,
            param_grid = self.param_grid,
            scoring = self.scoring,
            cv = self.cv,
            n_jobs = self.n_jobs,
            verbose = self.verbose,
            refit = True,
            error_score = "raise",
            return_train_score = False,
        )
        self._gs.fit(X, y)
        return SearchResult(
            algorithm = self.estimator.__class__.__name__,
            best_params = {k.replace("est__", ""): v for k, v in self._gs.best_params_.items()},
            best_score = self._gs.best_score_,
            scoring = self.scoring,
            cv = self.cv,
        )

    @property
    def best_estimator_(self):
        return self._gs.best_estimator_ if self._gs else None

    @property
    def best_params_(self):
        if not self._gs:
            return {}
        return {k.replace("est__", ""): v for k, v in self._gs.best_params_.items()}

    @property
    def best_score_(self):
        return self._gs.best_score_ if self._gs else None