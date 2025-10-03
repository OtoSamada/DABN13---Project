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


# Data split structure
@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None


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
        self.data = data if data is not None else self._load()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.metadata = ModelMetadata(experiment_name)
        
        if features is None:
            self.features = [c for c in self.data.columns if c != self.target_col]
        else:
            self.features = features  

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
        Create train/val/test split.
        val_size is proportion of the full dataset (not of train).
        """
        X = self.data[self.features].copy()
        y = self.data[self.target_col].astype(int)
        strat = y if self.stratify else None

        X_train_full, X_test, y_train_full, y_test = (
            train_test_split(
                X,
                y,
                test_size = self.test_size,
                random_state = self.random_state,
                stratify = strat
            )
        )

        if val_size > 0:
            relative_val = val_size / (1 - self.test_size)
            strat2 = y_train_full if self.stratify else None
            
            X_train, X_val, y_train, y_val = (
                train_test_split(
                    X_train_full,
                    y_train_full,
                    test_size = relative_val,
                    random_state = self.random_state,
                    stratify = strat2,
                )
            )

            self.split = (
                DataSplit(
                    X_train = X_train,
                    X_test = X_test,
                    y_train = y_train,
                    y_test = y_test,
                    X_val = X_val,
                    y_val = y_val,
                )
            )
            print(
                f"Split: train={len(X_train)} val={len(X_val)} test={len(X_test)} | "
                f"Positive Class proportions: Train={y_train.mean():.3f} Val={y_val.mean():.3f} Test={y_test.mean():.3f}"
            )

        else:
            self.split = (
                DataSplit(
                    X_train = X_train_full, 
                    X_test = X_test, 
                    y_train = y_train_full, 
                    y_test = y_test
                )
            )

            print(
                f"Split: train={len(X_train_full)} test={len(X_test)} | "
                f"Positive Class proportions: Train={y_train_full.mean():.3f} Test={y_test.mean():.3f}"
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


class HotelCancellationClassification(HotelAnalysis):
    """
    Main Class for training classification models on hotel cancellation data.
    Provides:
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

    # Grid Search Compatible with sklearn / xgboost style estimators
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
        Implement grid search using ModelSearch.
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

        search = (
            ModelSearch(
                estimator = estimator,
                param_grid = param_grid,
                numeric_features = numeric_features,
                categorical_features = categorical_features,
                scoring = scoring,
                cv = cv,
            )
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
        
        # Store with BOTH val_ and test_ prefixes for compatibility
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

        if eval_label == "val":
            metrics.update({
                "test_accuracy": metrics["val_accuracy"],
                "test_precision": metrics["val_precision"],
                "test_recall": metrics["val_recall"],
                "test_f1": metrics["val_f1"],
                "test_roc_auc": metrics["val_roc_auc"],
                "test_avg_precision": metrics["val_avg_precision"],
            })
        
        metrics["classification_report"] = (
            classification_report(
                eval_y, 
                preds, 
                output_dict = True
            )
        )

        self.metadata.log_experiment(
            algorithm = f"{result.algorithm}(GridSearch)",
            hyperparameters = result.best_params,
            results = metrics,
            data_info = self._data_info(),
            extra = {"cv": cv, "scoring": scoring},
        )

        key = f"{result.algorithm.lower()}_grid"
        self._persist_model(search.best_estimator_, key, scaler = None)
        return key

    
    def _compute_metrics(
        self,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        y_train_proba=None,
        y_test_proba=None,
    ) -> Dict[str, float]:
        """
        Compute standard classification metrics.
        """
        
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred, zero_division = 0),
            "train_recall": recall_score(y_train, y_train_pred, zero_division = 0),
            "train_f1": f1_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, zero_division = 0),
            "test_recall": recall_score(y_test, y_test_pred, zero_division = 0),
            "test_f1": f1_score(y_test, y_test_pred),
        }

        if y_train_proba is not None:
            metrics["train_roc_auc"] = roc_auc_score(y_train, y_train_proba)
            metrics["train_avg_precision"] = average_precision_score(y_train, y_train_proba)
        
        if y_test_proba is not None:
            metrics["test_roc_auc"] = roc_auc_score(y_test, y_test_proba)
            metrics["test_avg_precision"] = average_precision_score(y_test, y_test_proba)
        return metrics

    def _persist_model(
        self, model, key: str, scaler: Optional[StandardScaler] = None
    ) -> None:
        """Save trained model to models/trained/"""
        # Use absolute path from project root
        project_root = Path(__file__).parent.parent.parent
        out_dir = project_root / "models" / "trained"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if scaler is not None:
            obj = {"model": model, "scaler": scaler, "features": self.features}
        else:
            obj = {"model": model, "features": self.features}
        
        filepath = out_dir / f"{key}.joblib"
        joblib.dump(obj, filepath)
        print(f"Model saved: {filepath}")

    #  Visualization methods
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
        
        # Determine which model file to load based on algorithm
        if "GridSearch" in algo:
            # Strip (GridSearch) suffix for file pattern
            base_algo = algo.replace("(GridSearch)", "").strip().lower()
            pattern = f"{base_algo}_grid*.joblib"

        elif "Logistic" in algo:
            pattern = "logreg*.joblib"

        else:
            pattern = "dt_depth*.joblib"

        # Fetch the model from saved files
        saved = list(Path("models/trained").glob(pattern))

        if not saved:
            print(f"Model file not found matching: {pattern}")
            print(f"Available files: {list(Path('models/trained').glob('*.joblib'))}")
            return

        # Load model
        saved_path = max(saved, key = lambda p: p.stat().st_mtime)
        print(f"Loading model: {saved_path}")
        
        bundle = joblib.load(saved_path)
        model = bundle["model"]
        scaler = bundle.get("scaler")
        
        X_test = self.split.X_test
        y_test = self.split.y_test

        # Handle both pipeline and standalone models
        if hasattr(model, 'predict_proba'):
            # Model is a pipeline or standalone estimator
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Shouldn't reach here, but safety fallback
            print("Error: Model doesn't have predict_proba")
            return

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Evaluation Report: {algo}", fontsize=16, y=0.995)

        # 1. Confusion Matrix (top-left)
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax = axes[0, 0],
            cmap = "Blues",
            colorbar = False,
        )
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].grid(False)

        # 2. ROC Curve (top-right)
        RocCurveDisplay.from_predictions(
            y_test,
            y_proba,
            ax = axes[0, 1],
            name = algo,
            plot_chance_level = True,
        )
        axes[0, 1].set_title("ROC Curve")
        axes[0, 1].grid(alpha = 0.3)

        # 3. Precision-Recall Curve (bottom-left)
        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_proba,
            ax = axes[1, 0],
            name = algo,
        )
        axes[1, 0].set_title("Precision-Recall Curve")
        axes[1, 0].grid(alpha = 0.3)

        # 4. Model Metrics (bottom-right) - for all models
        metrics = experiment["results"]
        
        # Handle both val_ and test_ prefixes
        def get_metric(name):
            return metrics.get(f"test_{name}") or metrics.get(f"val_{name}") or 0
        
        metric_text = "\n".join(
            [
                f"Accuracy:  {get_metric('accuracy'):.4f}",
                f"Precision: {get_metric('precision'):.4f}",
                f"Recall:    {get_metric('recall'):.4f}",
                f"F1 Score:  {get_metric('f1'):.4f}",
                f"ROC-AUC:   {get_metric('roc_auc'):.4f}",
                f"Avg Prec:  {get_metric('avg_precision'):.4f}",
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
        axes[1, 1].set_title("Model Performance Metrics")
        axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            project_root = Path(__file__).parent.parent.parent
            abs_save_path = project_root / save_path
            abs_save_path.parent.mkdir(parents = True, exist_ok = True)
            plt.savefig(abs_save_path, dpi = 300, bbox_inches = "tight")
            print(f"Evaluation report saved: {abs_save_path}")

        plt.show()

    def display_feature_importance(
        self,
        experiment: Optional[Dict[str, Any]] = None,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """Display feature importance for tree-based models."""
        
        if experiment is None:
            dt_experiments = [
                e for e in self.metadata.metadata_log
                if "DecisionTree" in e["algorithm"]
            ]

            if not dt_experiments:
                print("No decision tree experiments found.")
                return
            experiment = dt_experiments[-1]

        algo = experiment["algorithm"]
        
        # Check if it's a tree-based model
        tree_models = ["DecisionTree", "RandomForest", "XGB", "GradientBoosting"]
        if not any(tree_model in algo for tree_model in tree_models):
            print(f"Feature importance only available for tree models. Got: {algo}")
            return

        # Try to get feature importance from results
        importance_dict = experiment["results"].get("feature_importance")
        
        # If not in results, try to load model
        if not importance_dict:
            if "GridSearch" in algo:
                base_algo = algo.replace("(GridSearch)", "").strip().lower()
                pattern = f"{base_algo}_grid*.joblib"
            else:
                pattern = f"{algo.lower()}*.joblib"
            
            saved = list(Path("models/trained").glob(pattern))
            
            if not saved:
                print(f"No model file found matching: {pattern}")
                return
            
            saved_path = max(saved, key=lambda p: p.stat().st_mtime)
            bundle = joblib.load(saved_path)
            model = bundle["model"]
            
            # Extract actual model from pipeline
            actual_model = model
            if hasattr(model, 'named_steps'):
                actual_model = model.named_steps.get('classifier', model.steps[-1][1])
            
            # Get feature importances
            if not hasattr(actual_model, 'feature_importances_'):
                print(f"Model doesn't have feature_importances_ attribute")
                return
            
            importances = actual_model.feature_importances_
            
            # Get feature names
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                try:
                    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                except:
                    feature_names = self.features
            else:
                feature_names = self.features
            
            importance_dict = dict(zip(feature_names, importances))

        if not importance_dict:
            print("No feature importance data found.")
            return

        # Sort and get top N features
        sorted_features = (
            sorted(
                importance_dict.items(), 
                key = lambda x: x[1], 
                reverse = True
            )
            [:top_n]
        )
        features, importances = zip(*sorted_features)

        # Create figure with two subplots
        fig = plt.figure(figsize=(figsize[0], figsize[1]))
        gs = fig.add_gridspec(2, 1, height_ratios = [3, 1], hspace = 0.3)
        
        # Main bar plot
        ax1 = fig.add_subplot(gs[0])
        y_pos = np.arange(len(features))
        bars = ax1.barh(y_pos, importances, color = 'steelblue', edgecolor = 'navy', linewidth = 0.5)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, importances)):
            ax1.text(val, i, f' {val:.4f}', va = 'center', fontsize = 8)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features, fontsize = 9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance', fontsize = 11, fontweight = 'bold')
        ax1.set_title(f'Top {top_n} Feature Importances\n{algo}', fontsize = 13, fontweight = 'bold', pad = 15)
        ax1.grid(axis = 'x', alpha = 0.3, linestyle = '--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Summary table
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        total_top = sum(importances)
        total_all = sum(importance_dict.values())
        coverage = total_top / total_all * 100
        
        summary_data = [
            ['Total Features', f"{len(importance_dict)}"],
            ['Top Features Shown', f"{len(features)}"],
            ['Top Features Sum', f"{total_top:.4f}"],
            ['Coverage', f"{coverage:.1f}%"]
        ]
        
        table = ax2.table(
            cellText = summary_data,
            colLabels = ['Metric', 'Value'],
            cellLoc = 'center',
            loc = 'center',
            bbox = [0.3, 0, 0.4, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        plt.tight_layout()
        plt.show()

        print(f"Feature Importance Analysis: {algo}")
        
        cumulative = 0
        for i, (feat, imp) in enumerate(sorted_features, 1):
            cumulative += imp
            cumulative_pct = (cumulative / total_all) * 100
            print(f"{i:<6} {feat:<40} {imp:<12.4f} {cumulative_pct:>6.1f}%")
        
        print(f"{'Total':<6} {'':<40} {total_all:<12.4f} {'100.0%':>6}")
        print(f"\nTop {top_n} features explain {coverage:.1f}% of total importance")

    def plot_tree(
        self,
        experiment: Optional[Dict[str, Any]] = None,
        tree_index: int = 0,
        max_depth: Optional[int] = None,
        figsize: Tuple[int, int] = (20, 10),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot decision tree structure for DecisionTree or XGBoost models.
        
        Args:
            experiment: Experiment dict from metadata. If None, uses last tree experiment.
            tree_index: For XGBoost, which tree to plot (0 to n_estimators-1)
            max_depth: Legacy parameter for finding DecisionTree by depth
            figsize: Figure size
            save_path: Path to save figure
        """
        # Determine which experiment to use
        if experiment is None and max_depth is not None:
            # Legacy: find DecisionTree by depth
            pattern = f"dt_depth_{max_depth if max_depth is not None else 'None'}*.joblib"
            saved = list(Path("models/trained").glob(pattern))
            if not saved:
                print(f"Decision tree model not found with pattern: {pattern}")
                return
            bundle = joblib.load(saved[0])
            model = bundle["model"]
            algo = "DecisionTreeClassifier"

        elif experiment is not None:
            # Use provided experiment
            algo = experiment["algorithm"]
            
            # Load model file
            if "GridSearch" in algo:
                base_algo = algo.replace("(GridSearch)", "").strip().lower()
                pattern = f"{base_algo}_grid*.joblib"
            elif "DecisionTree" in algo:
                pattern = "dt_depth*.joblib"
            elif "XGB" in algo:
                pattern = "xgbclassifier_grid*.joblib"
            else:
                pattern = f"{algo.lower()}*.joblib"
            
            saved = list(Path("models/trained").glob(pattern))
            if not saved:
                print(f"Model file not found matching: {pattern}")
                return
            
            saved_path = max(saved, key=lambda p: p.stat().st_mtime)
            bundle = joblib.load(saved_path)
            model = bundle["model"]
        else:
            print("Provide either experiment dict or max_depth parameter")
            return
        
        # Extract the actual model from pipeline if needed
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps['classifier']
        else:
            actual_model = model
        
        # Plot based on model type
        if "DecisionTree" in algo:
            # Sklearn DecisionTree
            plt.figure(figsize=figsize)
            plot_tree(
                actual_model,
                feature_names = self.features,
                class_names = ["Not Canceled", "Canceled"],
                filled = True,
                rounded = True,
                fontsize = 9,
            )
            plt.title(f"Decision Tree Structure\n{algo}")
            
        elif "XGB" in algo:
            # XGBoost - requires graphviz
            try:
                from xgboost import plot_tree as xgb_plot_tree
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=figsize)
                xgb_plot_tree(
                    actual_model,
                    num_trees = tree_index,
                    ax = ax,
                    rankdir = 'LR',
                )
                plt.title(f"XGBoost Tree #{tree_index}\n{algo}\n(Total trees: {actual_model.n_estimators})")
                
            except ImportError:
                print("XGBoost tree plotting requires graphviz.")
                print("Install with: brew install graphviz")
                print("Then: pip install graphviz")
                
                # Fallback: show feature importance instead
                print("\nShowing feature importance instead:")
                self.display_feature_importance(experiment=experiment, top_n=20)
                return
                
        else:
            print(f"Tree plotting not supported for: {algo}")
            return
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents = True, exist_ok = True)
            plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
            print(f"Tree plot saved: {save_path}")
        
        plt.show()