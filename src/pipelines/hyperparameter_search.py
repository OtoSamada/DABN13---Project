from __future__ import annotations
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class SearchResult:
    algorithm: str
    best_params: Dict[str, Any]
    best_score: float
    cv_results: Dict[str, Any]
    cv: int

class ModelSearch:
    def __init__(
        self,
        estimator,
        param_grid: Dict[str, List],
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scoring: str = "f1",
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        self.estimator = estimator
        self.param_grid = {f"est__{k}": v for k, v in param_grid.items()}
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._gs = None
        
        # Build preprocessing pipeline
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self) -> Pipeline:
        """Build preprocessing + estimator pipeline"""
        transformers = []
        
        # Add numeric transformer if numeric features exist
        if self.numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, self.numeric_features))
        
        # Add categorical transformer if categorical features exist
        if self.categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, self.categorical_features))
        
        # Create pipeline based on whether we have preprocessing steps
        if transformers:
            # We have preprocessing to do
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'  # Keep other columns as-is
            )
            pipeline = Pipeline([
                ('prep', preprocessor),
                ('est', self.estimator)
            ])
        else:
            # No preprocessing needed - data is already processed
            pipeline = Pipeline([
                ('est', self.estimator)
            ])
        
        return pipeline
    
    def fit(self, X: pd.DataFrame, y) -> SearchResult:
        self._gs = GridSearchCV(
            estimator = self.pipeline,
            param_grid = self.param_grid,
            scoring = self.scoring,
            cv = self.cv,
            n_jobs = self.n_jobs,
            verbose = self.verbose,
            refit = True,
            return_train_score = False,
        )
        self._gs.fit(X, y)
        return SearchResult(
            algorithm = self.estimator.__class__.__name__,
            best_params = {k.replace("est__", ""): v for k, v in self._gs.best_params_.items()},
            best_score = self._gs.best_score_,
            cv_results = self._gs.cv_results_,
            cv = self.cv,
        )
    
    def predict(self, X: pd.DataFrame):
        if self._gs is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self._gs.predict(X)
    
    def predict_proba(self, X: pd.DataFrame):
        if self._gs is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self._gs.predict_proba(X)
    
    @property
    def best_estimator_(self):
        if self._gs is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self._gs.best_estimator_

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