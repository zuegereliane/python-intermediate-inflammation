"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize('data, expected_standard_deviation, raises_assertion_error', [
    ([0, 0, 0], 0.0, False),
    ([1.0, 1.0, 1.0], 0, False),
    ([0.0, 2.0], 1.0, False),
    ([0.0, 2.0], 0.0, True),
    ([[0.0, 2.0], [1.0, 1.0]], [0.25, 0.25], False),
    ([[-1.0, -1.0]], [0., 0.], False)
])
def test_daily_standard_deviation(data, expected_standard_deviation, raises_assertion_error):
    from inflammation.models import daily_standard_deviation
    result_data = daily_standard_deviation(data)

    if raises_assertion_error:
        with pytest.raises(AssertionError):
            npt.assert_array_equal(np.array(result_data), np.array(expected_standard_deviation))
    else:
        npt.assert_array_equal(np.array(result_data), np.array(expected_standard_deviation))
