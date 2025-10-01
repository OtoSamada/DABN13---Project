from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    average_precision_score,
    ConfusionMatrixDisplay,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.utils.track_metadata import ModelMetadata
from src.pipelines.hyperparameter_search import ModelSearch


# --------------------------------------------------------------------------------------
# DATA SPLIT STRUCTURE
# --------------------------------------------------------------------------------------
@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None


# --------------------------------------------------------------------------------------
# BASE ANALYSIS CLASS
# --------------------------------------------------------------------------------------
class HotelAnalysis:
    """
    Base class: loads data, selects features, creates train/val/test splits.
    """

    def __init__(
        self,
        data_path: Path,
        data: Optional[pd.DataFrame] = None,
        target_col: str = "is_canceled",
        features: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        experiment_name: str = "hotel_cancellation",
    ):
        if data is None and data_path is None:
            raise ValueError("Provide either data (DataFrame) or data_path.")
        self.data_path = data_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.metadata = ModelMetadata(experiment_name)
        self.data = data if data is not None else self._load()
        self.features = features or [c for c in self.data.columns if c != self.target_col]
        self._validate()
        self.split: Optional[DataSplit] = None

    def _load(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)

    def _validate(self) -> None:
        missing = [c for c in self.features if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found.")

    def prepare_split(self, val_size: float = 0.0) -> None:
        """
        Create train/(val)/test split.
        val_size is proportion of the full dataset (not of train).
        """
        X = self.data[self.features].copy()
        y = self.data[self.target_col].astype(int)
        strat = y if self.stratify else None

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat,
        )

        if val_size > 0:
            relative_val = val_size / (1 - self.test_size)
            strat2 = y_train_full if self.stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=relative_val,
                random_state=self.random_state,
                stratify=strat2,
            )
            self.split = DataSplit(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                X_val=X_val,
                y_val=y_val,
            )
            print(
                f"Split: train={len(X_train)} val={len(X_val)} test={len(X_test)} | "
                f"TrainPos={y_train.mean():.3f} ValPos={y_val.mean():.3f} TestPos={y_test.mean():.3f}"
            )
        else:
            self.split = DataSplit(
                X_train=X_train_full, X_test=X_test, y_train=y_train_full, y_test=y_test
            )
            print(
                f"Split: train={len(X_train_full)} test={len(X_test)} | "
                f"TrainPos={y_train_full.mean():.3f} TestPos={y_test.mean():.3f}"
            )

    def _data_info(self) -> Dict[str, Any]:
        return {
            "n_rows": len(self.data),
            "n_features": len(self.features),
            "features": self.features,
            "target": self.target_col,
            "class_balance": self.data[self.target_col]
            .value_counts(normalize=True)
            .to_dict(),
        }


# --------------------------------------------------------------------------------------
# CLASSIFICATION ANALYSIS
# --------------------------------------------------------------------------------------
class HotelCancellationClassification(HotelAnalysis):
    """
    Provides:
      - Manual training for Logistic Regression, Decision Tree
      - Unified grid search for any sklearn / xgboost style estimator via ModelSearch
      - Comprehensive evaluation plots (confusion matrix, ROC, PR curves)
      - Feature importance visualization
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Identify numeric vs categorical features
        self.numeric_features = [
            c for c in self.features if self.data[c].dtype in ['int64', 'float64']
        ]
        self.categorical_features = [
            c for c in self.features if self.data[c].dtype == 'object'
        ]
        print(f"Features: {len(self.numeric_features)} numeric, {len(self.categorical_features)} categorical")

    def train_logistic_regression(
        self,
        C: float = 1.0,
        max_iter: int = 500,
        random_state: int = 42,
        scale: bool = True,
    ) -> str:
        if self.split is None:
            raise RuntimeError("Call prepare_split first.")
        
        X_train, X_test, y_train, y_test = (
            self.split.X_train,
            self.split.X_test,
            self.split.y_train,
            self.split.y_test,
        )

        # Only scale numeric features
        if scale and self.numeric_features:
            scaler = StandardScaler()
            X_train_numeric = scaler.fit_transform(X_train[self.numeric_features])
            X_test_numeric = scaler.transform(X_test[self.numeric_features])
            
            # Combine scaled numeric + categorical (if any)
            if self.categorical_features:
                X_train_cat = X_train[self.categorical_features].values
                X_test_cat = X_test[self.categorical_features].values
                X_train_t = np.hstack([X_train_numeric, X_train_cat])
                X_test_t = np.hstack([X_test_numeric, X_test_cat])
            else:
                X_train_t = X_train_numeric
                X_test_t = X_test_numeric
        else:
            scaler = None
            X_train_t = X_train.values
            X_test_t = X_test.values

        model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, n_jobs=-1)
        model.fit(X_train_t, y_train)

        train_pred = model.predict(X_train_t)
        test_pred = model.predict(X_test_t)
        train_proba = model.predict_proba(X_train_t)[:, 1]
        test_proba = model.predict_proba(X_test_t)[:, 1]

        results = self._compute_metrics(
            y_train, train_pred, y_test, test_pred, train_proba, test_proba
        )
        results["classification_report"] = classification_report(
            y_test, test_pred, output_dict=True
        )

        self.metadata.log_experiment(
            algorithm="LogisticRegression",
            hyperparameters={"C": C, "max_iter": max_iter, "scaled": scale},
            results=results,
            data_info=self._data_info(),
            random_state=random_state,
        )
        key = f"logreg_C{C}"
        self._persist_model(model, key, scaler)
        return key

    def train_decision_tree(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
    ) -> str:
        if self.split is None:
            raise RuntimeError("Call prepare_split first.")
        X_train, X_test, y_train, y_test = (
            self.split.X_train,
            self.split.X_test,
            self.split.y_train,
            self.split.y_test,
        )

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        results = self._compute_metrics(
            y_train, train_pred, y_test, test_pred, train_proba, test_proba
        )
        results["feature_importance"] = dict(
            zip(self.features, model.feature_importances_)
        )
        results["classification_report"] = classification_report(
            y_test, test_pred, output_dict=True
        )

        self.metadata.log_experiment(
            algorithm="DecisionTreeClassifier",
            hyperparameters={
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
            },
            results=results,
            data_info=self._data_info(),
            random_state=random_state,
        )
        key = f"dt_depth_{max_depth if max_depth is not None else 'None'}"
        self._persist_model(model, key)
        return key

    # --------- Unified grid search ----------
    def grid_search(
        self,
        estimator,
        param_grid: Dict[str, List[Any]],
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scoring: str = "f1",
        cv: int = 5,
        use_validation: bool = True,
    ) -> str:
        """
        Generic grid search using ModelSearch.
        Saves full preprocessing + estimator pipeline as a single artifact.

        estimator: any sklearn-compatible estimator (e.g., LogisticRegression, DecisionTreeClassifier, XGBClassifier)
        param_grid: dict of hyperparameters (no pipeline prefixes)
        numeric_features / categorical_features: optional overrides
        use_validation: evaluate on validation if available else test
        """
        if self.split is None:
            raise RuntimeError("Call prepare_split first.")

        if numeric_features is None:
            numeric_features = [
                c for c in self.features if self.data[c].dtype != "object"
            ]
        if categorical_features is None:
            categorical_features = [
                c for c in self.features if self.data[c].dtype == "object"
            ]

        search = ModelSearch(
            estimator=estimator,
            param_grid=param_grid,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scale_numeric=estimator.__class__.__name__.lower().startswith("logistic"),
            scoring=scoring,
            cv=cv,
        )
        X_train = self.split.X_train
        y_train = self.split.y_train
        result = search.fit(X_train, y_train)

        # Choose evaluation split
        if use_validation and self.split.X_val is not None:
            eval_X, eval_y = self.split.X_val, self.split.y_val
            eval_label = "val"
        else:
            eval_X, eval_y = self.split.X_test, self.split.y_test
            eval_label = "test"

        preds = search.best_estimator_.predict(eval_X)
        proba = search.best_estimator_.predict_proba(eval_X)[:, 1]
        
        metrics = {
            f"{eval_label}_accuracy": accuracy_score(eval_y, preds),
            f"{eval_label}_precision": precision_score(eval_y, preds, zero_division=0),
            f"{eval_label}_recall": recall_score(eval_y, preds, zero_division=0),
            f"{eval_label}_f1": f1_score(eval_y, preds),
            f"{eval_label}_roc_auc": roc_auc_score(eval_y, proba),
            f"{eval_label}_avg_precision": average_precision_score(eval_y, proba),
            "best_cv_score": result.best_score,
            "cv_scoring": scoring,
        }
        metrics["classification_report"] = classification_report(
            eval_y, preds, output_dict=True
        )

        self.metadata.log_experiment(
            algorithm=f"{result.algorithm}(GridSearch)",
            hyperparameters=result.best_params,
            results=metrics,
            data_info=self._data_info(),
            extra={"cv": cv, "scoring": scoring},
        )

        key = f"{result.algorithm.lower()}_grid"
        self._persist_model(search.best_estimator_, key, scaler=None)
        return key

    # --------- Utilities ----------
    def _compute_metrics(
        self,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        y_train_proba=None,
        y_test_proba=None,
    ) -> Dict[str, float]:
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
            "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
            "train_f1": f1_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_test_pred),
        }
        if y_train_proba is not None:
            metrics["train_roc_auc"] = roc_auc_score(y_train, y_train_proba)
            metrics["train_avg_precision"] = average_precision_score(
                y_train, y_train_proba
            )
        if y_test_proba is not None:
            metrics["test_roc_auc"] = roc_auc_score(y_test, y_test_proba)
            metrics["test_avg_precision"] = average_precision_score(y_test, y_test_proba)
        return metrics

    def _persist_model(
        self, model, key: str, scaler: Optional[StandardScaler] = None
    ) -> None:
        """Save trained model to models/trained/"""
        out_dir = Path("models/trained")
        out_dir.mkdir(parents=True, exist_ok=True)
        if scaler is not None:
            obj = {"model": model, "scaler": scaler, "features": self.features}
        else:
            obj = {"model": model, "features": self.features}
        joblib.dump(obj, out_dir / f"{key}.joblib")

    # --------- Visualization methods ----------
    def plot_evaluation_report(
        self,
        experiment: Dict[str, Any],
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Comprehensive evaluation plot with:
        - Confusion Matrix (native display)
        - ROC Curve (native display)
        - Precision-Recall Curve (native display)
        - Feature Importance (if tree-based) or Metrics Table

        2x2 grid layout.
        """
        if self.split is None:
            print("No split available.")
            return

        algo = experiment["algorithm"]
        key_part = "logreg" if "Logistic" in algo else "dt_depth"
        saved = list(Path("models/trained").glob(f"{key_part}*.joblib"))

        if not saved:
            print(f"Model file not found matching: {key_part}")
            return

        bundle = joblib.load(saved[0])
        model = bundle["model"]
        scaler = bundle.get("scaler")
        X_test = self.split.X_test
        y_test = self.split.y_test

        X_t = scaler.transform(X_test) if scaler else X_test.values
        y_pred = model.predict(X_t)
        y_proba = model.predict_proba(X_t)[:, 1]

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Evaluation Report: {algo}", fontsize=16, y=0.995)

        # 1. Confusion Matrix (top-left)
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=axes[0, 0],
            cmap="Blues",
            colorbar=False,
        )
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].grid(False)

        # 2. ROC Curve (top-right)
        RocCurveDisplay.from_predictions(
            y_test,
            y_proba,
            ax=axes[0, 1],
            name=algo,
            plot_chance_level=True,
        )
        axes[0, 1].set_title("ROC Curve")
        axes[0, 1].grid(alpha=0.3)

        # 3. Precision-Recall Curve (bottom-left)
        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_proba,
            ax=axes[1, 0],
            name=algo,
        )
        axes[1, 0].set_title("Precision-Recall Curve")
        axes[1, 0].grid(alpha=0.3)

        # 4. Feature Importance (bottom-right) - only for tree models
        if "DecisionTree" in algo:
            importance_dict = experiment["results"].get("feature_importance", {})
            if importance_dict:
                sorted_features = sorted(
                    importance_dict.items(), key=lambda x: x[1], reverse=True
                )[:15]
                features, importances = zip(*sorted_features)
                y_pos = np.arange(len(features))
                axes[1, 1].barh(y_pos, importances, color="steelblue")
                axes[1, 1].set_yticks(y_pos)
                axes[1, 1].set_yticklabels(features)
                axes[1, 1].invert_yaxis()
                axes[1, 1].set_xlabel("Importance")
                axes[1, 1].set_title("Top 15 Feature Importances")
                axes[1, 1].grid(axis="x", alpha=0.3)
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No feature importance data",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].axis("off")
        else:
            # For non-tree models, show metrics table
            metrics = experiment["results"]
            metric_text = "\n".join(
                [
                    f"Test Accuracy:  {metrics.get('test_accuracy', 0):.4f}",
                    f"Test Precision: {metrics.get('test_precision', 0):.4f}",
                    f"Test Recall:    {metrics.get('test_recall', 0):.4f}",
                    f"Test F1:        {metrics.get('test_f1', 0):.4f}",
                    f"Test ROC-AUC:   {metrics.get('test_roc_auc', 0):.4f}",
                    f"Test Avg Prec:  {metrics.get('test_avg_precision', 0):.4f}",
                ]
            )
            axes[1, 1].text(
                0.1,
                0.5,
                metric_text,
                ha="left",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=12,
                family="monospace",
            )
            axes[1, 1].set_title("Test Set Metrics")
            axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Evaluation report saved: {save_path}")

        plt.show()

    def display_feature_importance(
        self,
        experiment: Optional[Dict[str, Any]] = None,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """Display feature importance for tree-based models."""
        if experiment is None:
            dt_experiments = [
                e
                for e in self.metadata.metadata_log
                if "DecisionTree" in e["algorithm"]
            ]
            if not dt_experiments:
                print("No decision tree experiments found.")
                return
            experiment = dt_experiments[-1]

        algo = experiment["algorithm"]
        if "DecisionTree" not in algo:
            print(f"Feature importance only available for tree models. Got: {algo}")
            return

        importance_dict = experiment["results"].get("feature_importance")
        if not importance_dict:
            print("No feature importance data found.")
            return

        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        features, importances = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances\n{algo}")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"\nFeature Importance Summary ({algo}):")
        print("-" * 50)
        for feat, imp in sorted_features:
            print(f"{feat:30s}: {imp:.4f}")

        total_top = sum(importances)
        total_all = sum(importance_dict.values())
        print("-" * 50)
        print(f"Total (top {top_n}): {total_top:.4f}")
        print(f"Coverage: {total_top/total_all*100:.1f}% of total importance")

    def plot_tree(
        self,
        max_depth: Optional[int] = None,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot decision tree structure."""
        pattern = f"dt_depth_{max_depth if max_depth is not None else 'None'}*.joblib"
        saved = list(Path("models/trained").glob(pattern))
        if not saved:
            print("Decision tree model not found.")
            return
        bundle = joblib.load(saved[0])
        model: DecisionTreeClassifier = bundle["model"]
        plt.figure(figsize=figsize)
        plot_tree(
            model,
            feature_names=self.features,
            class_names=["Not Canceled", "Canceled"],
            filled=True,
            rounded=True,
            fontsize=9,
        )
        plt.title(f"Decision Tree (depth={max_depth})")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


# --------------------------------------------------------------------------------------
# EXPERIMENT RUNNER
# --------------------------------------------------------------------------------------
def run_hotel_classification_experiments(
    data_path: str | Path,
    logistic_C: List[float],
    tree_depths: List[Optional[int]],
    metric: str = "test_f1",
    test_size: float = 0.2,
    random_state: int = 42,
    val_size: float = 0.0,
    use_grid: bool = False,
    save_metadata: bool = True,
    save_best: bool = True,
) -> HotelCancellationClassification:
    """
    Brute-force loop OR grid search (if use_grid=True).
    Returns analysis object with metadata populated.

    If use_grid=True:
      - logistic_C and tree_depths serve as param grids.
    """
    analysis = HotelCancellationClassification(
        data_path=Path(data_path),
        test_size=test_size,
        random_state=random_state,
        experiment_name="hotel_cancellation",
    )
    analysis.prepare_split(val_size=val_size)

    if use_grid:
        print("\n[GridSearch] Logistic Regression")
        analysis.grid_search(
            estimator=LogisticRegression(max_iter=1000, random_state=random_state),
            param_grid={"C": logistic_C},
            scoring="f1",
            cv=5,
            use_validation=True,
        )
        print("[GridSearch] Decision Tree")
        analysis.grid_search(
            estimator=DecisionTreeClassifier(random_state=random_state),
            param_grid={"max_depth": tree_depths, "min_samples_split": [2, 5, 10]},
            scoring="f1",
            cv=5,
            use_validation=True,
        )
    else:
        for C in logistic_C:
            analysis.train_logistic_regression(C=C, random_state=random_state)
        for depth in tree_depths:
            analysis.train_decision_tree(max_depth=depth, random_state=random_state)

    best = analysis.metadata.best(metric=metric)
    if best:
        print(
            f"\nBest overall ({metric}): {best['algorithm']} "
            f"{best['results'].get(metric, best['results'].get('test_f1')):.4f} "
            f"params={best['hyperparameters']}"
        )

    if save_metadata:
        analysis.metadata.save()

    if save_best:
        analysis.metadata.save_best_model(metric=metric)

    analysis.metadata.summary()
    return analysis