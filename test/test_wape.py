import numpy as np
import pytest

from dwabi.wape import observed_support, wape_deciles, wape_point


def test_cell_volume_matches_inverse_weighted_uniform_median():
    deciles = np.arange(1., 10.).reshape(1, 1, 1, 9)
    point = np.full((1, 1, 1), 5.)

    result = wape_point(point, deciles, epsilon=0.)

    # Continuous mass between 1 and 9, with flat 10% tails at each end.
    expected = np.exp(0.5 * np.log(9.) + 0.5 / 9. - 0.5)
    np.testing.assert_allclose(result, expected, rtol=0., atol=1e-9)
    np.testing.assert_array_equal(point, 5.)


def test_support_projects_each_target_after_volume_action():
    point = np.array([[[2.5, -1.4, 1.5]]], dtype=np.float32)
    deciles = np.repeat(point[..., None], 9, axis=-1)
    result = wape_point(
        point, deciles, epsilon=0., nonnegative=[True, True, False],
        integer=[True, False, True],
    )
    np.testing.assert_array_equal(result, [[[2., 0., 2.]]])
    assert result.dtype == point.dtype


def test_quantile_pool_keeps_the_original_q50():
    raw = np.array([4., 3., 2., 1., 0., 9., 8., 7., 6.]).reshape(1, 1, 1, 1, 9)

    calculus, served = wape_deciles(raw)

    np.testing.assert_array_equal(calculus[0, 0, 0], [0., 1., 2., 3., 4., 6., 7., 8., 9.])
    np.testing.assert_array_equal(served[0, 0, 0], [0., 0., 0., 0., 0., 6., 7., 8., 9.])


def test_observed_support_excludes_all_nonfinite_values():
    lower, integer = observed_support([
        [0., 1., None, np.nan],
        [-2., 1., np.inf],
        [0., 3.25, np.nan],
        [None, np.inf, -np.inf],
        [],
    ])
    np.testing.assert_array_equal(lower, [True, False, True, False, False])
    np.testing.assert_array_equal(integer, [True, True, False, False, False])
    assert lower.dtype == integer.dtype == np.bool_


def test_rejects_unsorted_deciles():
    deciles = np.arange(9., 0., -1).reshape(1, 1, 1, 9)
    with pytest.raises(ValueError, match="ordered"):
        wape_point(np.ones((1, 1, 1)), deciles, epsilon=0.)

