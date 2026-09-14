import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from ggslicer import suggest_contour_levels, slice_contours


def test_quantile_method_returns_exactly_n_interior_quantiles():
    x = np.arange(1, 101, dtype=float)
    df = pd.DataFrame({"value": x})
    levels = suggest_contour_levels(df, method="quantile", n=5)
    expected = np.quantile(x, np.arange(1, 6) / 6)
    np.testing.assert_allclose(levels, expected)
    assert len(levels) == 5


def test_quantile_method_respects_custom_n():
    x = np.arange(1, 101, dtype=float)
    df = pd.DataFrame({"value": x})
    levels = suggest_contour_levels(df, method="quantile", n=3)
    assert len(levels) == 3


def test_troughs_method_finds_real_boundary_of_bimodal_distribution():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        bimodal = np.concatenate([
            rng.normal(0, 1, 1000),
            rng.normal(20, 1, 1000),
        ])
        df = pd.DataFrame({"value": bimodal})
        levels = suggest_contour_levels(df, method="troughs", min_n=1)
        assert np.all((levels > 3) & (levels < 17)), f"seed {seed}: {levels}"


def test_troughs_method_pads_unimodal_distribution_to_exactly_min_n():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        unimodal = rng.normal(0, 1, 1000)
        df = pd.DataFrame({"value": unimodal})
        levels = suggest_contour_levels(df, method="troughs", min_n=3)
        assert len(levels) == 3


def test_troughs_method_finds_multiple_boundaries_for_trimodal_distribution():
    rng = np.random.default_rng(1)
    trimodal = np.concatenate([
        rng.normal(0, 1, 1000),
        rng.normal(20, 1, 1000),
        rng.normal(40, 1, 1000),
    ])
    df = pd.DataFrame({"value": trimodal})
    levels = suggest_contour_levels(df, method="troughs", min_n=1)
    assert len(levels) >= 2
    assert np.any((levels > 3) & (levels < 17))
    assert np.any((levels > 23) & (levels < 37))


def test_dataframe_and_image_and_path_inputs_agree(make_image, tmp_path):
    rng = np.random.default_rng(1)
    img = make_image((10, 10, 10), lambda idx: float(sum(idx)) + rng.uniform())

    levels_img = suggest_contour_levels(img, method="quantile", n=3)
    assert len(levels_img) == 3

    path = str(tmp_path / "img.nii.gz")
    sitk.WriteImage(img, path)
    levels_path = suggest_contour_levels(path, method="quantile", n=3)
    np.testing.assert_allclose(levels_img, levels_path)


def test_column_name_matching_discrete_data_names_errors_clearly():
    df = pd.DataFrame({"mask": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="discrete"):
        suggest_contour_levels(df, column="mask")


def test_data_driven_discreteness_check_catches_oddly_named_discrete_column():
    df = pd.DataFrame({"oddname": np.tile(np.arange(4), 25)})
    with pytest.raises(ValueError, match="discrete"):
        suggest_contour_levels(df, column="oddname")


def test_genuinely_continuous_oddly_named_column_is_not_flagged_as_discrete():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"oddname": rng.normal(0, 1, 1000)})
    suggest_contour_levels(df, column="oddname")


def test_suggest_contour_levels_validates_its_arguments():
    df = pd.DataFrame({"value": np.random.default_rng(1).normal(0, 1, 100)})
    with pytest.raises(ValueError):
        suggest_contour_levels(df, method="bogus")
    with pytest.raises(ValueError):
        suggest_contour_levels(df, n=0)
    with pytest.raises(ValueError):
        suggest_contour_levels(df, n=-1)
    with pytest.raises(ValueError):
        suggest_contour_levels(df, method="troughs", min_n=0)
    with pytest.raises(ValueError, match="no column"):
        suggest_contour_levels(df, column="missing")


def test_suggested_levels_work_directly_as_slice_contours_levels(make_image):
    def value_fn(idx):
        return float(np.hypot(idx[0] - 10, idx[1] - 10))

    img = make_image((20, 20, 20), value_fn)
    levels = suggest_contour_levels(img, method="quantile", n=3)
    out = slice_contours(img, axis="z", coordinate=10, levels=levels)
    assert len(out) > 0
