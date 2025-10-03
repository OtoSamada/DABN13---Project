from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PreprocessingConfig:
    """
    Configuration for hotel booking preprocessing.

    Holds parameters and learned state for preprocessing steps.
    """
    target_col: str = "is_canceled"

    # Continent grouping
    continent_groups: Dict[str, set] = field(default_factory=lambda: {
        'Europe': {
            'PRT','GBR','ESP','IRL','FRA','ROU','NOR','POL','DEU','BEL','CHE','GRC','ITA','NLD','DNK','SWE','EST',
            'CZE','FIN','LUX','SVN','ALB','UKR','SMR','LVA','SRB','AUT','BLR','LTU','TUR','HRV','AND','GIB','SVK',
            'HUN','BIH','BGR','MKD','ISL','MLT','IMN','LIE','MNE','FRO','MCO','CYP','JEY','GGY'
        },
        'North America': {
            'USA','CAN','MEX','PRI','JAM','CYM','KNA','GLP','BRB','DMA','VGB','CRI','CUB','DOM','GTM','HND','LCA',
            'NIC','PAN','SLV','AIA','ABW','BHS'
        },
        'South America': {'BRA','CHL','COL','VEN','SUR','ECU','PER','BOL','PRY','GUY','ARG','URY'},
        'Asia': {
            'CHN','CN','IND','KOR','HKG','IRN','ARE','GEO','ARM','ISR','PHL','IDN','JPN','KWT','MDV','THA','MYS','LKA',
            'SGP','MMR','UZB','KAZ','SAU','VNM','TJK','PAK','NPL','BGD','QAT','MAC','IRQ','JOR','LAO','LBN','OMN',
            'BHR','AZE','TMP','TWN'
        },
        'Africa': {
            'MOZ','MAR','AGO','ZAF','EGY','NGA','KEN','ZWE','DZA','TUN','CMR','CIV','COM','UGA','GAB','GNB','MRT','DJI',
            'STP','TZA','SDN','LBR','BDI','BEN','BFA','CPV','ETH','GHA','MDG','MLI','MUS','MWI','MYT','NAM','SEN','SLE',
            'SYC','TGO','RWA','LBY'
        },
        'Oceania': {'AUS','FJI','NZL','PNG','PLW','NCL','KIR','FSM','SLB','NRU','TUV','TON','WSM','ASM','UMI'},
        'Others': {'ATA','ATF','UNK'}
    })

    # Columns to drop
    drop_columns: Tuple[str, ...] = (
        "reservation_status",
        "reservation_status_date",
        "assigned_room_type",
        "arrival_date_year",
        "agent",
        "company",
    )

    # Quantile-capped columns & percentile
    quantile_cap_cols: Tuple[str, ...] = (
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "days_in_waiting_list",
        "adr",
    )
    quantile_p: float = 0.99

    # Hard thresholds
    adr_upper_threshold: float = 1000.0
    stays_in_week_nights_max: int = 30
    adults_max: int = 6

    # Transform targets
    log_transform_cols: Tuple[str, ...] = ("lead_time", "stays_in_week_nights", "adr")
    zero_inflated_cols: Tuple[str, ...] = (
        "children",
        "babies",
        "days_in_waiting_list",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "required_car_parking_spaces",
        "stays_in_weekend_nights",
    )

    # Binning specifications for high cardinality data
    adults_bins: Tuple[float, ...] = (0, 1, 2, np.inf)
    adults_labels: Tuple[str, ...] = ("1", "2", "3+")

    special_req_bins: Tuple[float, ...] = (-0.1, 0, 1, np.inf)
    special_req_labels: Tuple[str, ...] = ("0", "1", "2+")

    weekend_nights_bins: Tuple[float, ...] = (-0.1, 0, 1, 2, np.inf)
    weekend_nights_labels: Tuple[str, ...] = ("0", "1", "2", "3+")

    # Cyclical encoding for week number column
    week_cycle_mod: int = 53

    # Output
    output_dir: Path | str = "../../data/processed"

    # Runtime flags
    verbose: bool = False

    # Feature selection (base + engineered)
    base_keep: Tuple[str, ...] = (
        "hotel",
        "lead_time",
        "adults",
        "children",
        "babies",
        "meal",
        "distribution_channel",
        "is_repeated_guest",
        "reserved_room_type",
        "deposit_type",
        "customer_type",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "days_in_waiting_list",
        "continent",
        "arrival_date_month",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "adr",
    )

    engineered_keep: Tuple[str, ...] = (
        "lead_time_log",
        "stays_in_week_nights_log",
        "adr_log",
        "children_gt0",
        "babies_gt0",
        "days_in_waiting_list_gt0",
        "previous_cancellations_gt0",
        "previous_bookings_not_canceled_gt0",
        "required_car_parking_spaces_gt0",
        "stays_in_weekend_nights_gt0",
        "adults_cat",
        "total_of_special_requests_cat",
        "stays_in_weekend_nights_cat",
        "arrival_week_sin",
        "arrival_week_cos",
    )

    # Generic imputation toggle
    learn_generic_imputers: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HotelPreprocessor:
    """
    A preprocessing class, replicating steps taken from the exploratory notebook.
    Adds learned medians/modes + quantile caps for inference robustness.
    """

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        **overrides: Any,
    ) -> None:
        
        self.config = config or PreprocessingConfig()
        
        # Apply any overrides to the original configuration
        for _config, _value in overrides.items():
            if hasattr(self.config, _config):
                setattr(self.config, _config, _value)
            else:
                raise AttributeError(f"Unknown config field: {_config}")

        # Learned metrics, useful for infrence
        self._quantile_caps: Dict[str, float] = {}
        self._adr_median: Optional[float] = None
        self._numeric_medians: Dict[str, float] = {}
        self._categorical_modes: Dict[str, Any] = {}
        self._training_columns: List[str] = []
        self._fitted: bool = False

        # Pre-compute continent lookup
        self._continent_lookup: Dict[str, str] = {
            code: cont for cont, codes in self.config.continent_groups.items() for code in codes
        }

    # Public API
    def fit(self, df: pd.DataFrame) -> "HotelPreprocessor":
        self._validate_input(df)
        tmp = df.copy()

        # Track original columns
        self._training_columns = list(tmp.columns)

        # Get ADR median (ignore negatives)
        if "adr" in tmp.columns and tmp["adr"].notna().any():
            self._adr_median = float(tmp.loc[tmp["adr"] >= 0, "adr"].median())

        # Quantile caps
        for col in self.config.quantile_cap_cols:
            if col in tmp.columns and tmp[col].notna().any():
                self._quantile_caps[col] = float(tmp[col].quantile(self.config.quantile_p))

        # Generic imputers, learn medians and modes to impute missing values
        if self.config.learn_generic_imputers:
            # Exclude target & columns slated for drop
            drop_set = set(self.config.drop_columns)
            target = self.config.target_col
            numeric_cols = [
                c for c in tmp.select_dtypes(include = [np.number]).columns
                if c != target and c not in drop_set
            ]
            cat_cols = [
                c for c in tmp.select_dtypes(include = ["object", "category"]).columns
                if c != target and c not in drop_set
            ]
            for c in numeric_cols:
                if tmp[c].notna().any():
                    self._numeric_medians[c] = float(tmp[c].median())
            for c in cat_cols:
                mode_vals = tmp[c].mode(dropna = True)
                if not mode_vals.empty:
                    self._categorical_modes[c] = mode_vals.iloc[0]

        self._fitted = True
        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        
        self._validate_input(df)
        cfg = self.config
        df_proc = df.copy()

        # 1. Drop duplicates
        df_proc = df_proc.drop_duplicates().reset_index(drop = True)

        # 2. Drop configured columns
        to_drop = [c for c in cfg.drop_columns if c in df_proc.columns]
        if to_drop:
            df_proc = df_proc.drop(columns=to_drop)

        # 3. Country -> continent (drop missing country rows)
        if "country" in df_proc.columns:
            df_proc = df_proc[~df_proc["country"].isna()].copy()
            df_proc["continent"] = df_proc["country"].map(self._continent_lookup).fillna("Others")
            df_proc = df_proc.drop(columns = ["country"])

        # 4. Children missing -> 0
        if "children" in df_proc.columns:
            df_proc["children"] = df_proc["children"].fillna(0)

        # 5. Noise cleaning
        if "adr" in df_proc.columns and self._adr_median is not None:
            df_proc.loc[df_proc["adr"] < 0, "adr"] = self._adr_median

        if "adults" in df_proc.columns:
            df_proc = df_proc[(df_proc["adults"] != 0) & (df_proc["adults"] <= cfg.adults_max)]
        
        for col in ("children", "babies"):
            if col in df_proc.columns:
                df_proc = df_proc[df_proc[col] < 10]
        
        if "stays_in_week_nights" in df_proc.columns:
            df_proc = df_proc[df_proc["stays_in_week_nights"] <= cfg.stays_in_week_nights_max]

        # 6. Quantile caps
        for col, cap in self._quantile_caps.items():
            if col in df_proc.columns:
                if col == "adr":
                    df_proc.loc[df_proc[col] > cfg.adr_upper_threshold, col] = cap
                df_proc.loc[df_proc[col] > cap, col] = cap

        # 7. Generic imputation (median/mode)
        if cfg.learn_generic_imputers:
            for c, val in self._numeric_medians.items():
                if c in df_proc.columns:
                    df_proc[c] = df_proc[c].fillna(val)

            for c, val in self._categorical_modes.items():
                if c in df_proc.columns:
                    df_proc[c] = df_proc[c].fillna(val)

        # 8. Arrival day grouping
        if "arrival_date_day_of_month" in df_proc.columns:
            day = df_proc["arrival_date_day_of_month"]
            df_proc["arrival_date_day_of_month"] = (
                np.select(
                    [day < 11, day < 21],
                    ["BoM", "MoM"],
                    default = "EoM",
                )            
            )

        # 9. Log transforms
        for col in cfg.log_transform_cols:
            if col in df_proc.columns:
                df_proc[f"{col}_log"] = np.log1p(df_proc[col].clip(lower = 0))

        # 10. Zero-inflated flags
        for col in cfg.zero_inflated_cols:
            if col in df_proc.columns:
                df_proc[f"{col}_gt0"] = (df_proc[col] > 0).astype(np.uint8)

        # 11. Binning
        if "adults" in df_proc.columns:
            df_proc["adults_cat"] = (
                pd.cut(
                    df_proc["adults"],
                    bins = cfg.adults_bins,
                    labels = cfg.adults_labels,
                    include_lowest = True
                )
                .astype(str)
            )
        if "total_of_special_requests" in df_proc.columns:
            df_proc["total_of_special_requests_cat"] = (
                pd.cut(
                    df_proc["total_of_special_requests"],
                    bins = cfg.special_req_bins,
                    labels = cfg.special_req_labels,
                    include_lowest = True
                )
                .astype(str)
            )

        if "stays_in_weekend_nights" in df_proc.columns:
            df_proc["stays_in_weekend_nights_cat"] = (
                pd.cut(
                    df_proc["stays_in_weekend_nights"],
                    bins = cfg.weekend_nights_bins,
                    labels = cfg.weekend_nights_labels,
                    include_lowest = True
                )
                .astype(str)
            )

        # 12. Cyclical encoding
        if "arrival_date_week_number" in df_proc.columns:
            w = df_proc["arrival_date_week_number"].fillna(0).astype(float)
            mod = cfg.week_cycle_mod
            df_proc["arrival_week_sin"] = np.sin(2 * np.pi * w / mod)
            df_proc["arrival_week_cos"] = np.cos(2 * np.pi * w / mod)

        # 13. Warning for missing expected columns
        missing_expected = [c for c in self._training_columns if c not in df_proc.columns]
        if missing_expected and cfg.verbose:
            print(f"Missing columns during transform: {missing_expected}")

        return df_proc.reset_index(drop = True)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def transform_one(self, row: Dict[str, Any]) -> pd.DataFrame:
        """Convenience: transform a single raw record dict."""
        return self.transform(pd.DataFrame([row]))

    def select_columns(
        self,
        df: pd.DataFrame,
        include_engineered: bool = True
    ) -> pd.DataFrame:
        cfg = self.config
        cols = [cfg.target_col] + list(cfg.base_keep)
        if not include_engineered:
            kept = [c for c in cols if c in df.columns]
            return df[kept].copy()
        engineered = [c for c in cfg.engineered_keep if c in df.columns]
        ordered = []
        seen = set()
        for c in cols + engineered:
            if c in df.columns and c not in seen:
                seen.add(c)
                ordered.append(c)
        return df[ordered].copy()

    def save(self, df: pd.DataFrame, filename: str = "hotel_bookings_processed.csv") -> Path:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        df.to_csv(path, index=False)
        return path

    # Internals / properties
    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

    @property
    def quantile_caps_(self) -> Dict[str, float]:
        return dict(self._quantile_caps)

    @property
    def numeric_medians_(self) -> Dict[str, float]:
        return dict(self._numeric_medians)

    @property
    def categorical_modes_(self) -> Dict[str, Any]:
        return dict(self._categorical_modes)

    @property
    def fitted_(self) -> bool:
        return self._fitted

    @property
    def config_dict(self) -> Dict[str, Any]:
        return self.config.as_dict()