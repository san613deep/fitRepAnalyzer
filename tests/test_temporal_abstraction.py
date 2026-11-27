"""Tests for rolling-window sensor feature calculations."""
import numpy as np
import pandas as pd

from src.features.TemporalAbstraction import NumericalAbstraction


def test_rolling_mean_uses_only_the_current_and_previous_window():
    source = pd.DataFrame({"acc_x": [1.0, 2.0, 3.0, 4.0]})

    result = NumericalAbstraction().abstract_numerical(
        source.copy(), ["acc_x"], window_size=2, aggregation_function="mean"
    )

    values = result["acc_x_temp_mean_ws_2"].to_numpy()
    assert np.isnan(values[0])
    np.testing.assert_allclose(values[1:], [1.5, 2.5, 3.5])


def test_rolling_std_is_zero_for_constant_window():
    source = pd.DataFrame({"gyr_z": [2.0, 2.0, 2.0]})

    result = NumericalAbstraction().abstract_numerical(
        source.copy(), ["gyr_z"], window_size=2, aggregation_function="std"
    )

    assert np.isnan(result.loc[0, "gyr_z_temp_std_ws_2"])
    np.testing.assert_allclose(
        result.loc[1:, "gyr_z_temp_std_ws_2"].to_numpy(), [0.0, 0.0]
    )
