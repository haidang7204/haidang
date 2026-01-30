"""
preprocessing.py

Reusable preprocessing module for the Porto Seguro safe driver project.
Implements the InsurancePreprocessor class used in the notebook pipeline.
"""

from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


class InsurancePreprocessor:
    """
    Modular preprocessing pipeline for insurance risk modeling.

    This class handles:
    - Missing value imputation (median for continuous, mode for categoricals)
    - Target encoding for high-cardinality categorical features
    - One-hot encoding for low-cardinality categorical features
    - Standard scaling for continuous features
    - Optional interaction and missing-indicator features (already in X)
    - Low-variance feature removal

    Parameters
    ----------
    continuous_cols : list of str
        Names of continuous (interval) features.
    binary_cols : list of str
        Names of binary features (0/1 or _bin columns).
    low_card_cat_cols : list of str
        Categorical features with low cardinality to be one-hot encoded.
    high_card_cat_cols : list of str
        Categorical features with high cardinality to be target encoded.
    extra_feature_cols : list of str, optional
        Any additional numeric features already in X (e.g., interaction terms,
        missing indicators) that should simply be passed through.
    variance_threshold : float, default 0.01
        Threshold for sklearn.feature_selection.VarianceThreshold.
    smoothing : float, default 1.0
        Smoothing factor for target encoding.
    """

    def __init__(
        self,
        continuous_cols: List[str],
        binary_cols: List[str],
        low_card_cat_cols: List[str],
        high_card_cat_cols: List[str],
        extra_feature_cols: Optional[List[str]] = None,
        variance_threshold: float = 0.01,
        smoothing: float = 1.0,
    ) -> None:
        self.continuous_cols = continuous_cols
        self.binary_cols = binary_cols
        self.low_card_cat_cols = low_card_cat_cols
        self.high_card_cat_cols = high_card_cat_cols
        self.extra_feature_cols = extra_feature_cols or []
        self.variance_threshold = variance_threshold
        self.smoothing = smoothing

        # Learned objects
        self.median_imputer_cont: Optional[SimpleImputer] = None
        self.mode_imputer_cat: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.variance_selector: Optional[VarianceThreshold] = None

        # Target encoding
        self.global_mean: Optional[float] = None
        self.target_encoding_maps: Dict[str, pd.Series] = {}

        # One-hot encoding
        self.oh_columns: Optional[List[str]] = None

        # Bookkeeping
        self.selected_columns: Optional[List[str]] = None
        self.fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self.fitted:
            raise ValueError("InsurancePreprocessor is not fitted. Call fit() first.")

    def _target_encode_column(
        self, train_df: pd.DataFrame, col: str, y: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Target-encode a single high-cardinality categorical column.

        Uses the smoothing formula from the project handout:
        smoothed = (count * mean + smoothing * global_mean) / (count + smoothing)
        """
        assert self.global_mean is not None

        agg_df = train_df.groupby(col)[y.name].agg(["count", "mean"])
        agg_df.columns = ["count", "mean"]
        agg_df["smoothed"] = (
            agg_df["count"] * agg_df["mean"] + self.smoothing * self.global_mean
        ) / (agg_df["count"] + self.smoothing)

        mapping = agg_df["smoothed"]
        self.target_encoding_maps[col] = mapping

        train_encoded = train_df[col].map(mapping).fillna(self.global_mean)
        return train_encoded, mapping

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "InsurancePreprocessor":
        """
        Learn all preprocessing parameters from the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training target (binary).

        Returns
        -------
        InsurancePreprocessor
            Fitted instance (for chaining).
        """
        X = X.copy()
        y = y.copy()

        # Global mean for target encoding
        self.global_mean = float(y.mean())

        # Split into groups
        cont_cols = [c for c in self.continuous_cols if c in X.columns]
        cat_cols = [
            c
            for c in (
                self.binary_cols
                + self.low_card_cat_cols
                + self.high_card_cat_cols
            )
            if c in X.columns
        ]
        extra_cols = [c for c in self.extra_feature_cols if c in X.columns]

        # ------------------ Imputation ------------------
        # Continuous: median
        if cont_cols:
            self.median_imputer_cont = SimpleImputer(strategy="median")
            X_cont = pd.DataFrame(
                self.median_imputer_cont.fit_transform(X[cont_cols]),
                columns=cont_cols,
                index=X.index,
            )
        else:
            X_cont = pd.DataFrame(index=X.index)

        # Categorical: mode
        if cat_cols:
            self.mode_imputer_cat = SimpleImputer(strategy="most_frequent")
            X_cat = pd.DataFrame(
                self.mode_imputer_cat.fit_transform(X[cat_cols]),
                columns=cat_cols,
                index=X.index,
            )
        else:
            X_cat = pd.DataFrame(index=X.index)

        # Extra features: just take them as-is
        X_extra = X[extra_cols] if extra_cols else pd.DataFrame(index=X.index)

        # ------------------ Target encoding ------------------
        # Apply to high-cardinality categorical columns
        for col in self.high_card_cat_cols:
            if col in X_cat.columns:
                encoded_col, _ = self._target_encode_column(
                    train_df=X_cat[[col, y.name]].join(y), col=col, y=y
                )
                X_cat[col + "_te"] = encoded_col
                X_cat.drop(columns=[col], inplace=True)

        # ------------------ One-hot encoding ------------------
        low_card_cols_present = [
            c for c in self.low_card_cat_cols if c in X_cat.columns
        ]
        if low_card_cols_present:
            X_low = pd.get_dummies(
                X_cat[low_card_cols_present], drop_first=True
            )
            self.oh_columns = list(X_low.columns)
        else:
            X_low = pd.DataFrame(index=X.index)
            self.oh_columns = []

        # Keep remaining categorical columns (e.g., binaries not one-hot encoded
        # and the new *_te columns)
        remaining_cat_cols = [
            c
            for c in X_cat.columns
            if c not in low_card_cols_present
        ]
        X_cat_remaining = X_cat[remaining_cat_cols]

        # ------------------ Scaling ------------------
        if cont_cols:
            self.scaler = StandardScaler()
            X_cont_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_cont),
                columns=cont_cols,
                index=X.index,
            )
        else:
            X_cont_scaled = X_cont

        # ------------------ Combine all features ------------------
        X_full = pd.concat(
            [X_cont_scaled, X_cat_remaining, X_low, X_extra],
            axis=1,
        )

        # ------------------ Variance threshold ------------------
        self.variance_selector = VarianceThreshold(
            threshold=self.variance_threshold
        )
        X_sel = self.variance_selector.fit_transform(X_full)
        support_mask = self.variance_selector.get_support()
        self.selected_columns = list(X_full.columns[support_mask])

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned preprocessing steps to new data.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform (train, validation, or test).

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix ready for modeling.
        """
        self._check_fitted()
        X = X.copy()

        cont_cols = [c for c in self.continuous_cols if c in X.columns]
        cat_cols = [
            c
            for c in (
                self.binary_cols
                + self.low_card_cat_cols
                + self.high_card_cat_cols
            )
            if c in X.columns
        ]
        extra_cols = [c for c in self.extra_feature_cols if c in X.columns]

        # ------------------ Imputation ------------------
        # Continuous
        if cont_cols and self.median_imputer_cont is not None:
            X_cont = pd.DataFrame(
                self.median_imputer_cont.transform(X[cont_cols]),
                columns=cont_cols,
                index=X.index,
            )
        else:
            X_cont = pd.DataFrame(index=X.index)

        # Categorical
        if cat_cols and self.mode_imputer_cat is not None:
            X_cat = pd.DataFrame(
                self.mode_imputer_cat.transform(X[cat_cols]),
                columns=cat_cols,
                index=X.index,
            )
        else:
            X_cat = pd.DataFrame(index=X.index)

        # Extra features
        X_extra = X[extra_cols] if extra_cols else pd.DataFrame(index=X.index)

        # ------------------ Target encoding ------------------
        if self.global_mean is None:
            raise ValueError("global_mean is None. Fit the preprocessor first.")

        for col in self.high_card_cat_cols:
            if col in X_cat.columns and col in self.target_encoding_maps:
                mapping = self.target_encoding_maps[col]
                X_cat[col + "_te"] = X_cat[col].map(mapping).fillna(
                    self.global_mean
                )
                X_cat.drop(columns=[col], inplace=True)

        # ------------------ One-hot encoding ------------------
        low_card_cols_present = [
            c for c in self.low_card_cat_cols if c in X_cat.columns
        ]
        if low_card_cols_present:
            X_low = pd.get_dummies(
                X_cat[low_card_cols_present], drop_first=True
            )
        else:
            X_low = pd.DataFrame(index=X.index)

        # Align dummy columns to training-time columns
        if self.oh_columns is not None:
            for col in self.oh_columns:
                if col not in X_low.columns:
                    X_low[col] = 0
            extra_dummy_cols = [
                c for c in X_low.columns if c not in self.oh_columns
            ]
            if extra_dummy_cols:
                X_low.drop(columns=extra_dummy_cols, inplace=True)
            X_low = X_low[self.oh_columns]

        remaining_cat_cols = [
            c
            for c in X_cat.columns
            if c not in low_card_cols_present
        ]
        X_cat_remaining = X_cat[remaining_cat_cols]

        # ------------------ Scaling ------------------
        if cont_cols and self.scaler is not None:
            X_cont_scaled = pd.DataFrame(
                self.scaler.transform(X_cont),
                columns=cont_cols,
                index=X.index,
            )
        else:
            X_cont_scaled = X_cont

        # ------------------ Combine and select ------------------
        X_full = pd.concat(
            [X_cont_scaled, X_cat_remaining, X_low, X_extra],
            axis=1,
        )

        if self.variance_selector is None or self.selected_columns is None:
            raise ValueError(
                "Variance selector not initialized. Fit the preprocessor first."
            )

        X_sel_array = self.variance_selector.transform(X_full)
        X_sel = pd.DataFrame(
            X_sel_array, columns=self.selected_columns, index=X.index
        )

        return X_sel

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Convenience method: fit the preprocessor on X, y and
        return the transformed X.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training target.

        Returns
        -------
        pd.DataFrame
            Transformed training features.
        """
        self.fit(X, y)
        return self.transform(X)
