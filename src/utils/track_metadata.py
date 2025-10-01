from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelMetadata:
    """
    Collects and persists experiment runs (append-merge).
    Each record:
      - experiment_name
      - timestamp (UTC ISO)
      - algorithm
      - hyperparameters
      - results (metrics)
      - data_info (shape, features, etc.)
      - random_state
      - optional extra (like best_cv_score)
    
    Paths:
      - Metadata JSON: models/experiments/<experiment_name>.json
      - Trained models: models/trained/
      - Best models: models/best_models/
    """

    def __init__(self, experiment_name: str = "hotel_cancellation"):
        self.experiment_name = experiment_name
        self.metadata_log: List[Dict[str, Any]] = []

    def log_experiment(
        self,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        results: Dict[str, Any],
        data_info: Dict[str, Any],
        random_state: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.utcnow().isoformat(),
            "algorithm": algorithm,
            "hyperparameters": hyperparameters,
            "results": results,
            "random_state": random_state,
            "data_info": data_info,
        }
        if extra:
            rec.update(extra)
        self.metadata_log.append(rec)

    def _key(self, meta: Dict[str, Any]) -> str:
        return (
            f"{meta['algorithm']}|"
            f"{json.dumps(meta['hyperparameters'], sort_keys=True)}|"
            f"{meta.get('random_state')}"
        )

    def save(self, path: Optional[str] = None) -> None:
        """Save metadata JSON to models/experiments/<experiment_name>.json"""
        if path is None:
            path = f"models/experiments/{self.experiment_name}.json"
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if p.exists() and p.stat().st_size > 0:
            try:
                existing = json.loads(p.read_text())
            except Exception:
                existing = []
        merged = {self._key(m): m for m in existing}
        for m in self.metadata_log:
            merged[self._key(m)] = m
        p.write_text(json.dumps(list(merged.values()), indent=2))
        print(f"Metadata written: {p} (total {len(merged)} experiments)")

    def best(self, metric: str = "test_f1", algorithm: Optional[str] = None) -> Optional[Dict[str, Any]]:
        runs = self.metadata_log
        if algorithm:
            runs = [r for r in runs if r["algorithm"] == algorithm]
        if not runs:
            return None
        return max(runs, key=lambda r: r["results"].get(metric, float("-inf")))

    def save_best_model(
        self,
        metric: str = "test_f1",
        source_dir: Path = Path("models/trained"),
        dest_dir: Path = Path("models/best_models"),
    ) -> None:
        """
        Copy best model artifact (by metric) to best_models subfolder.
        Naming convention: best_<algorithm_lower>.joblib
        """
        best = self.best(metric=metric)
        if not best:
            print("No experiments logged; cannot identify best model.")
            return

        algo = best["algorithm"].lower().replace("(gridsearch)", "").replace(" ", "_").strip()
        
        # Search for model file matching algorithm name
        patterns = [f"*{algo}*.joblib", f"{algo}*.joblib"]
        source_files = []
        for pat in patterns:
            source_files.extend(source_dir.glob(pat))

        if not source_files:
            print(f"No model file found in {source_dir} matching algorithm: {algo}")
            return

        # Pick most recent if multiple
        source_file = max(source_files, key=lambda p: p.stat().st_mtime)

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"best_{algo}.joblib"

        shutil.copy2(source_file, dest_file)
        print(f"Best model copied: {source_file.name} -> {dest_file}")
        print(f"  Metric: {metric} = {best['results'].get(metric, 'N/A'):.4f}")
        print(f"  Hyperparameters: {best['hyperparameters']}")

    def summary(self) -> None:
        print(f"\nEXPERIMENT SUMMARY [{self.experiment_name}]")
        print("-" * 70)
        for i, r in enumerate(self.metadata_log, 1):
            acc = r["results"].get("test_accuracy") or r["results"].get("val_accuracy") or r["results"].get("val_or_test_accuracy")
            f1 = r["results"].get("test_f1") or r["results"].get("val_f1") or r["results"].get("val_or_test_f1")
            print(
                f"{i}. {r['algorithm']} params={r['hyperparameters']} "
                f"metrics={{acc={acc:.3f if acc else 0:.3f}, f1={f1:.3f if f1 else 0:.3f}}}"
            )