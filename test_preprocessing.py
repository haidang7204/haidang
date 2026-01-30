# test_preprocessing.py

import numpy as np
import pandas as pd
import pytest

from preprocessing import InsurancePreprocessor


@pytest.fixture
def sample_data():
    """Small dummy dataset with the same column types you use."""
    df = pd.DataFrame({
        "ps_reg_01": [0.1, 0.2, np.nan, 0.4],   # continuous
        "ps_reg_02": [1.0, 1.0, 1.0, 1.0],      # near-constant continuous
        "ps_ind_06_bin": [0, 1, 0, 1],          # binary
        "ps_car_12": [2.0, np.nan, 3.5, 4.1],   # continuous
    })
    y = pd.Series([0, 1, 0, 1], name="target")

    continuous = ["ps_reg_01", "ps_reg_02", "ps_car_12"]
    binary = ["ps_ind_06_bin"]
    low_card = []
    high_card = []

    return df, y, continuous, binary, low_card, high_card


def test_transform_before_fit_raises(sample_data):
    """Calling transform before fit should raise an error."""
    df, y, continuous, binary, low_card, high_card = sample_data

    pre = InsurancePreprocessor(
        continuous_cols=continuous,
        binary_cols=binary,
        low_card_cat_cols=low_card,
        high_card_cat_cols=high_card,
    )

    with pytest.raises(ValueError):
        pre.transform(df)


def test_fit_transform_then_transform_shape_and_columns(sample_data):
    """
    After fitting, transform should:
      - keep the same number of rows
      - produce consistent columns between fit_transform and transform
    """
    df, y, continuous, binary, low_card, high_card = sample_data

    pre = InsurancePreprocessor(
        continuous_cols=continuous,
        binary_cols=binary,
        low_card_cat_cols=low_card,
        high_card_cat_cols=high_card,
    )

    X_clean = pre.fit_transform(df, y)

    # same number of rows as input
    assert X_clean.shape[0] == df.shape[0]

    # calling transform again should give same shape and column order
    X_clean_2 = pre.transform(df.copy())
    assert X_clean_2.shape == X_clean.shape
    assert list(X_clean_2.columns) == list(X_clean.columns)
