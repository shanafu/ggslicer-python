import numpy as np
import pytest
from plotnine import ggplot, aes, geom_raster

from ggslicer import (
    ReadImage_fix,
    slice_image,
    slice_label_contours,
    contour_centroids,
    slice_label_annotations,
    slice_label_annotations_layer,
)


def _center_coronal_coordinate(labels):
    size = labels.GetSize()
    center_idx = (size[0] // 2, size[1] // 2, size[2] // 2)
    return labels.TransformIndexToPhysicalPoint(center_idx)


def test_slice_label_annotations_on_mouse_1_real_dsurqe_atlas(testdata_dir):
    base = testdata_dir / "mouse_1"
    labels = ReadImage_fix(str(base / "DSURQE_40micron_labels.mnc"))
    center_world = _center_coronal_coordinate(labels)

    contours = slice_label_contours(labels, axis="coronal", coordinate=center_world[1])
    out = contour_centroids(contours)

    assert len(out) > 0
    contour_pairs = set(zip(contours["label"], contours["obj"]))
    out_pairs = set(zip(out["label"], out["obj"]))
    assert contour_pairs == out_pairs

    assert np.all(np.isfinite(out["x"])) and np.all(np.isfinite(out["y"])) and np.all(np.isfinite(out["z"]))
    assert np.all(out["area"] > 0)
    assert np.all(out["name"] == out["label"].apply(lambda v: str(int(round(v)))))


def test_min_area_meaningfully_reduces_real_cluttered_atlas_output(testdata_dir):
    base = testdata_dir / "mouse_1"
    labels = ReadImage_fix(str(base / "DSURQE_40micron_labels.mnc"))
    center_world = _center_coronal_coordinate(labels)

    all_out = slice_label_annotations(labels, axis="coronal", coordinate=center_world[1])
    filtered = slice_label_annotations(labels, axis="coronal", coordinate=center_world[1], min_area=1)

    assert len(filtered) < len(all_out)
    assert np.all(filtered["area"] >= 1)

    largest = all_out.loc[all_out["area"].idxmax()]
    match = filtered[(filtered["label"] == largest["label"]) & (filtered["obj"] == largest["obj"])]
    assert len(match) == 1


def test_label_names_correctly_annotates_real_dsurqe_label_ids(testdata_dir):
    base = testdata_dir / "mouse_1"
    labels = ReadImage_fix(str(base / "DSURQE_40micron_labels.mnc"))
    center_world = _center_coronal_coordinate(labels)

    contours = slice_label_contours(labels, axis="coronal", coordinate=center_world[1])
    present_labels = contours["label"].unique()
    if len(present_labels) < 2:
        pytest.skip("not enough distinct real labels at this slice to test naming")

    name_map = {int(round(lb)): f"Region_{int(round(lb))}" for lb in present_labels[:2]}
    out = contour_centroids(contours, label_names=name_map)

    for lb in present_labels[:2]:
        key = int(round(lb))
        assert np.all(out.loc[out["label"] == lb, "name"] == f"Region_{key}")


def test_slice_label_annotations_layer_renders_real_ggplot_over_template(testdata_dir):
    base = testdata_dir / "mouse_1"
    average = ReadImage_fix(str(base / "DSURQE_40micron_average.mnc"))
    labels = ReadImage_fix(str(base / "DSURQE_40micron_labels.mnc"))
    center_world = _center_coronal_coordinate(labels)

    img_df = slice_image(average, axis="coronal", coordinate=center_world[1])
    annotations = slice_label_annotations(labels, axis="coronal", coordinate=center_world[1], min_area=1)

    p = (
        ggplot()
        + geom_raster(data=img_df, mapping=aes(x="x", y="y", fill="value"))
        + slice_label_annotations_layer(annotations, color="white")
    )
    fig = p.draw()
    assert fig is not None
