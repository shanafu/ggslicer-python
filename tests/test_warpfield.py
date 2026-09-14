import numpy as np
import pytest
import SimpleITK as sitk
from plotnine import ggplot, aes, geom_segment

from ggslicer import SliceGeometry, SlicePackage, slice_warp_arrows, slice_warp_arrows_layer


def _make_image(size):
    return sitk.Image(list(size), sitk.sitkFloat32)


def _make_vector_image(size, value_fn):
    vec_img = sitk.Image(list(size), sitk.sitkVectorFloat64, 3)
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                vec_img.SetPixel((i, j, k), value_fn((i, j, k)))
    return vec_img


def _write_test_xfm(path, body_lines):
    with open(path, "w") as f:
        f.write("MNI Transform File\n%test\n\n")
        f.write("\n".join(body_lines) + "\n")


def test_constant_translation_warp_gives_expected_decomposition_on_axial_slice():
    image = _make_image((20, 20, 5))
    t = sitk.TranslationTransform(3, (3, 4, 5))

    arrows = slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=5)

    for col in ("package", "k", "i", "j", "x", "y", "z", "xend", "yend", "zend",
                "in_plane_displacement", "normal_displacement", "total_displacement"):
        assert col in arrows.columns

    assert np.allclose(arrows["in_plane_displacement"], 5)  # sqrt(3^2+4^2) = 5
    assert np.allclose(np.abs(arrows["normal_displacement"]), 5)
    assert np.allclose(
        arrows["total_displacement"],
        np.sqrt(arrows["in_plane_displacement"] ** 2 + arrows["normal_displacement"] ** 2),
    )

    assert np.allclose(arrows["xend"] - arrows["x"], 3)
    assert np.allclose(arrows["yend"] - arrows["y"], 4)
    assert np.allclose(arrows["zend"], arrows["z"])  # arrow never leaves the slice plane


def test_arrow_length_rescales_drawn_arrow_but_not_true_displacement():
    image = _make_image((20, 20, 5))
    t = sitk.TranslationTransform(3, (3, 4, 0))

    true_arrows = slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=5)
    scaled_arrows = slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=5, arrow_length=2)

    assert np.allclose(true_arrows["in_plane_displacement"], scaled_arrows["in_plane_displacement"])
    assert np.allclose(true_arrows["total_displacement"], scaled_arrows["total_displacement"])

    drawn_len = np.sqrt(
        (scaled_arrows["xend"] - scaled_arrows["x"]) ** 2
        + (scaled_arrows["yend"] - scaled_arrows["y"]) ** 2
        + (scaled_arrows["zend"] - scaled_arrows["z"]) ** 2
    )
    assert np.allclose(drawn_len, 2)

    true_dir = np.column_stack([true_arrows["xend"] - true_arrows["x"], true_arrows["yend"] - true_arrows["y"]])
    true_dir = true_dir / np.linalg.norm(true_dir, axis=1, keepdims=True)
    scaled_dir = np.column_stack([scaled_arrows["xend"] - scaled_arrows["x"], scaled_arrows["yend"] - scaled_arrows["y"]])
    scaled_dir = scaled_dir / np.linalg.norm(scaled_dir, axis=1, keepdims=True)
    assert np.allclose(scaled_dir, true_dir)


def test_purely_out_of_plane_displacement_leaves_zero_length_arrow_even_with_arrow_length():
    image = _make_image((20, 20, 5))
    t = sitk.TranslationTransform(3, (0, 0, 7))  # pure z (normal) displacement on an axial slice

    arrows = slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=5, arrow_length=3)

    assert np.allclose(arrows["in_plane_displacement"], 0)
    assert np.allclose(arrows["normal_displacement"], 7)
    assert np.allclose(arrows["xend"], arrows["x"])
    assert np.allclose(arrows["yend"], arrows["y"])
    assert np.allclose(arrows["zend"], arrows["z"])


def test_raw_vector_image_and_displacement_field_transform_give_identical_results():
    image = _make_image((10, 10, 3))

    def field(idx):
        return (idx[0] * 0.1, idx[1] * 0.2, 0.5)

    vec_img_for_image_form = _make_vector_image((10, 10, 3), field)
    arrows_image_form = slice_warp_arrows(image, axis="axial", coordinate=0, warp=vec_img_for_image_form, spacing=3)

    # sitk.DisplacementFieldTransform(image) moves/invalidates the source image
    # object (confirmed in this codebase; see warpfield.py and CLAUDE.md) --
    # build a fresh copy for the Transform-object input form.
    vec_img_for_transform_form = _make_vector_image((10, 10, 3), field)
    transform_obj = sitk.DisplacementFieldTransform(vec_img_for_transform_form)
    arrows_transform_form = slice_warp_arrows(image, axis="axial", coordinate=0, warp=transform_obj, spacing=3)

    assert np.allclose(arrows_image_form["total_displacement"], arrows_transform_form["total_displacement"])
    assert np.allclose(arrows_image_form["xend"], arrows_transform_form["xend"])


def test_warp_given_as_xfm_path_matches_equivalent_conjugated_transform_object(tmp_path):
    path = str(tmp_path / "t.xfm")
    _write_test_xfm(path, [
        "Transform_Type = Linear;",
        "Linear_Transform =",
        " 1 0 0 2",
        " 0 1 0 -1",
        " 0 0 1 0;",
    ])

    image = _make_image((10, 10, 3))
    # A .xfm path routes through read_minc_transform()'s default
    # corrected=True (see transform.py), which conjugates the file's raw
    # translation (2, -1, 0) by negating x/y: for a pure translation, this
    # is equivalent to just negating the translation vector's x/y
    # components, i.e. (-2, 1, 0).
    arrows_path = slice_warp_arrows(image, axis="axial", coordinate=0, warp=path, spacing=3)

    t = sitk.TranslationTransform(3, (-2, 1, 0))
    arrows_transform = slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=3)

    assert np.allclose(arrows_path["xend"], arrows_transform["xend"])
    assert np.allclose(arrows_path["yend"], arrows_transform["yend"])


def test_invert_applies_the_inverse_warp():
    image = _make_image((10, 10, 3))
    t = sitk.TranslationTransform(3, (3, 4, 0))

    forward = slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=5)
    inverted = slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=5, invert=True)

    assert np.allclose(inverted["xend"] - inverted["x"], -3)
    assert np.allclose(inverted["yend"] - inverted["y"], -4)
    assert np.allclose(forward["in_plane_displacement"], inverted["in_plane_displacement"])


def test_spacing_controls_lattice_density_and_default_matches_slice_grid_convention():
    image = _make_image((40, 40, 3))
    t = sitk.TranslationTransform(3, (1, 1, 0))

    coarse = slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=10)
    fine = slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=2)
    assert len(coarse) < len(fine)

    geom = SliceGeometry.from_image_axis(image, "axial", 0)
    default_arrows = slice_warp_arrows(geometry=geom, warp=t)
    extent = geom.extent
    expected_spacing = float(np.min(extent)) / 10
    expected_size = np.round(extent / expected_spacing).astype(int) + 1
    assert len(default_arrows) == int(np.prod(expected_size))


def test_slice_warp_arrows_supports_multi_slice_package_and_geometry_escape_hatch():
    image = _make_image((10, 10, 5))
    t = sitk.TranslationTransform(3, (1, 1, 1))
    pkg = SlicePackage.from_image_axis(image, "axial")

    arrows = slice_warp_arrows(geometry=pkg, warp=t, spacing=3)
    assert set(arrows["k"].unique()) == set(range(5))
    assert (arrows["package"] == "package_1").all()


def test_slice_warp_arrows_validates_its_arguments():
    image = _make_image((10, 10, 3))
    t = sitk.TranslationTransform(3, (1, 1, 1))

    with pytest.raises(ValueError):
        slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=-1)
    with pytest.raises(ValueError):
        slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, arrow_length=0)
    # falls through to sitk.ReadTransform(), which raises its own RuntimeError
    # for an unreadable path (SimpleITK's exception type, not this package's).
    with pytest.raises(RuntimeError):
        slice_warp_arrows(image, axis="axial", coordinate=0, warp="not_a_real_object")
    with pytest.raises(ValueError):
        slice_warp_arrows(
            image, axis="axial", coordinate=0, warp=t,
            geometry=SliceGeometry.from_image_axis(image, "axial", 0),
        )


def test_orthonormal_invariant_holds_on_an_oblique_slice():
    image = _make_image((20, 20, 20))

    def field(idx):
        return (idx[0] * 0.05, -idx[1] * 0.03, idx[2] * 0.02)

    vec_img = _make_vector_image((20, 20, 20), field)
    geom = SliceGeometry.from_normal(origin=(2, 2, 2), normal=(1, 1, 1), spacing=(1, 1), size=(5, 5))

    arrows = slice_warp_arrows(geometry=geom, warp=vec_img, spacing=1)
    inv_err = np.max(np.abs(
        arrows["total_displacement"] ** 2
        - (arrows["in_plane_displacement"] ** 2 + arrows["normal_displacement"] ** 2)
    ))
    assert inv_err < 1e-8


def test_slice_warp_arrows_layer_builds_a_usable_plotnine_layer():
    image = _make_image((10, 10, 3))
    t = sitk.TranslationTransform(3, (1, 1, 0))
    arrows = slice_warp_arrows(image, axis="axial", coordinate=0, warp=t, spacing=3)

    layer = slice_warp_arrows_layer(arrows)
    assert isinstance(layer, geom_segment)

    p = ggplot(arrows, aes(x="x", y="y", xend="xend", yend="yend")) + layer
    assert isinstance(p, ggplot)
