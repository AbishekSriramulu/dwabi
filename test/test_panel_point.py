import numpy as np
import polars as pl

from dwabi import DWabi


def member(point, median, *, include_point=True):
    def predict(request):
        values = {name: pl.lit(median) for name in request.quantile_cols}
        if include_point:
            values["point"] = pl.lit(point)
        return request.future.with_columns(**values).reverse()

    return predict


def test_wape_panel_fits_from_earlier_points_and_reorders_member_rows():
    train = pl.DataFrame({"unique_id": ["a"] * 30, "ds": range(30), "y": [10] * 30})
    models = {"low": member(2., 8.), "accurate": member(10., 50.), "high": member(40., 200.)}
    model = DWabi(models, h=2, objective="wape").fit(train)
    issued = model.predict()
    assert len(model.decision_.ordinal_origins) >= 3
    np.testing.assert_array_equal(issued["point"], [10., 10.])
    np.testing.assert_array_equal(issued["q0.5"], [50., 50.])


def test_wape_panel_accepts_median_only_members_and_insufficient_history():
    train = pl.DataFrame({"unique_id": ["a", "a"], "ds": [0, 1], "y": [4, 4]})
    models = {str(y): member(0., y, include_point=False) for y in (2., 4., 8.)}
    forecast = DWabi(models, h=2, objective="wape").fit(train).predict()
    np.testing.assert_array_equal(forecast["point"], [4., 4.])
