import numpy as np
import pytest

from ggslicer import SliceGeometry, SlicePackage, SlicePackageSet


def make_package(direction_i=(1, 0, 0), direction_j=(0, 1, 0), size=(3, 3), spacing_k=1, size_k=2):
    base = SliceGeometry(origin=[0, 0, 0], direction_i=direction_i, direction_j=direction_j, spacing=[1, 1], size=size)
    return SlicePackage(base_slice=base, spacing_k=spacing_k, size_k=size_k)


def test_constructor_validates_its_inputs():
    with pytest.raises(ValueError, match="SlicePackage"):
        SlicePackageSet({"a": 1, "b": 2})
    with pytest.raises(ValueError, match="non-empty string"):
        SlicePackageSet({"": make_package()})
    with pytest.raises(ValueError, match="dict"):
        SlicePackageSet("not a dict")
    SlicePackageSet()  # empty is fine
    SlicePackageSet({"a": make_package()})  # single is fine


def test_set_package_remove_package_package_names():
    sset = SlicePackageSet()
    assert sset.package_names == []

    sset.set_package("axial", make_package())
    assert sset.package_names == ["axial"]

    with pytest.raises(ValueError, match="SlicePackage"):
        sset.set_package("axial", "not a package")
    with pytest.raises(ValueError, match="non-empty string"):
        sset.set_package(1, make_package())

    sset.set_package("sagittal", make_package(direction_i=(0, 1, 0), direction_j=(0, 0, 1)))
    assert set(sset.package_names) == {"axial", "sagittal"}

    sset.remove_package("axial")
    assert sset.package_names == ["sagittal"]


def test_sample_points_combines_packages_with_a_package_column():
    sset = SlicePackageSet({
        "axial": make_package(size_k=4),
        "sagittal": make_package(direction_i=(0, 1, 0), direction_j=(0, 0, 1), size_k=2),
    })
    pts = sset.sample_points
    assert list(pts.columns)[0] == "package"
    assert set(pts["package"].unique()) == {"axial", "sagittal"}
    assert len(pts) == 3 * 3 * 4 + 3 * 3 * 2


def test_sample_points_on_an_empty_set_returns_a_typed_0_row_frame():
    sset = SlicePackageSet()
    pts = sset.sample_points
    assert len(pts) == 0
    assert list(pts.columns) == ["package", "i", "j", "k", "x", "y", "z"]


def test_sample_intensity_on_a_3d_image_with_no_extra_index(make_image):
    sset = SlicePackageSet({"only": make_package(size=(5, 5), size_k=10)})
    src = make_image((5, 5, 10), lambda idx: 100 * idx[2] + 10 * idx[0] + idx[1])

    out = sset.sample_intensity(src)
    assert len(out) == 5 * 5 * 10
    row = out[(out["i"] == 1) & (out["j"] == 2) & (out["k"] == 3)]
    assert row["intensity"].iloc[0] == 100 * 3 + 10 * 1 + 2


def test_sample_intensity_handles_a_4d_image_via_extract_and_cache(make_image):
    img4d = make_image((5, 5, 10, 3), lambda idx: 1000 * idx[3] + 100 * idx[2] + 10 * idx[0] + idx[1])

    sset = SlicePackageSet({
        "axial": make_package(size=(5, 5), size_k=10),
        "sagittal": make_package(direction_i=(0, 1, 0), direction_j=(0, 0, 1), size=(5, 10), size_k=5),
    })

    out = sset.sample_intensity(img4d, extra_index={"t": range(3)})

    assert len(out) == (5 * 5 * 10 + 5 * 10 * 5) * 3
    assert "package" in out.columns and "t" in out.columns

    chk = out[(out["package"] == "axial") & (out["t"] == 2) & (out["i"] == 2) & (out["j"] == 1) & (out["k"] == 3)]
    assert len(chk) == 1
    assert chk["intensity"].iloc[0] == img4d.GetPixel((2, 1, 3, 2))


def test_sample_intensity_handles_a_5d_image(make_image):
    img5d = make_image(
        (2, 2, 2, 2, 3),
        lambda idx: 1000 * idx[4] + 100 * idx[3] + 10 * idx[0] + idx[1] + idx[2],
    )
    sset = SlicePackageSet({"only": make_package(size=(2, 2), size_k=2)})

    out = sset.sample_intensity(img5d, extra_index={"t": range(2), "c": range(3)})

    assert len(out) == 2 * 2 * 2 * 2 * 3
    chk = out[(out["t"] == 1) & (out["c"] == 2) & (out["i"] == 1) & (out["j"] == 0) & (out["k"] == 1)]
    assert len(chk) == 1
    assert chk["intensity"].iloc[0] == img5d.GetPixel((1, 0, 1, 1, 2))


def test_sample_intensity_validates_image_dimension_against_extra_index(make_image):
    sset = SlicePackageSet({"only": make_package()})
    src3d = make_image((3, 3, 2), lambda idx: 0)
    img4d = make_image((2, 2, 2, 2), lambda idx: 0)

    with pytest.raises(ValueError, match="0 non-spatial"):
        sset.sample_intensity(src3d, extra_index={"t": range(2)})
    with pytest.raises(ValueError, match="1 non-spatial"):
        sset.sample_intensity(img4d)
    with pytest.raises(ValueError, match="1 non-spatial"):
        sset.sample_intensity(img4d, extra_index={"t": range(2), "c": range(2)})


def test_sample_intensity_errors_when_the_set_has_no_packages(make_image):
    sset = SlicePackageSet()
    src3d = make_image((3, 3, 2), lambda idx: 0)
    with pytest.raises(ValueError, match="No packages"):
        sset.sample_intensity(src3d)


def test_set_packages_bulk_replaces_the_whole_collection():
    sset = SlicePackageSet({"a": make_package()})
    sset.packages = {"x": make_package(), "y": make_package(size_k=3)}
    assert set(sset.package_names) == {"x", "y"}

    sset.packages = {}
    assert sset.package_names == []

    with pytest.raises(ValueError):
        sset.packages = {"a": 1, "b": 2}


def test_rename_package_renames_without_disturbing_the_package():
    pkg = make_package()
    sset = SlicePackageSet({"old": pkg})
    sset.rename_package("old", "new")

    assert sset.package_names == ["new"]
    assert sset.packages["new"] is pkg

    with pytest.raises(ValueError, match="No package named"):
        sset.rename_package("missing", "z")

    sset.set_package("another", make_package())
    with pytest.raises(ValueError, match="already exists"):
        sset.rename_package("new", "another")


def test_str_reports_each_packages_name_size_and_spacing():
    sset = SlicePackageSet({"axial": make_package(), "sagittal": make_package(size_k=3)})
    s = str(sset)
    assert "<SlicePackageSet>" in s
    assert "axial" in s
    assert "sagittal" in s
    assert "size = " in s
    assert "0 package" in str(SlicePackageSet())


def test_bare_slice_geometry_accepted_wherever_slice_package_expected():
    slice_ = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [3, 3])

    sset = SlicePackageSet({"single": slice_})
    wrapped = sset.packages["single"]
    assert isinstance(wrapped, SlicePackage)
    assert wrapped.size_k == 1
    assert np.allclose(wrapped.base_slice.origin, slice_.origin)

    sset.set_package("another", slice_)
    assert isinstance(sset.packages["another"], SlicePackage)

    sset.packages = {"mixed_a": slice_, "mixed_b": make_package()}
    assert all(isinstance(v, SlicePackage) for v in sset.packages.values())

    with pytest.raises(ValueError, match="SlicePackage.*SliceGeometry"):
        SlicePackageSet({"bad": "not a slice or package"})


def test_from_orthogonal_triplet_builds_the_classic_3_plane_view(make_image):
    img = make_image((10, 10, 10), lambda idx: 0, origin=(-5, -5, -5), spacing=(1, 1, 1))

    triplet = SlicePackageSet.from_orthogonal_triplet(img, {"x": 0, "coronal": 1, "horizontal": -2})
    assert set(triplet.package_names) == {"sagittal", "coronal", "axial"}
    assert triplet.packages["sagittal"].size_k == 1

    partial = SlicePackageSet.from_orthogonal_triplet(img, {"axial": 0})
    assert partial.package_names == ["axial"]

    with pytest.raises(ValueError, match="more than once"):
        SlicePackageSet.from_orthogonal_triplet(img, {"x": 0, "sagittal": 1})


def test_from_slice_packages_builds_a_set_from_an_unnamed_list_or_a_single_package():
    base = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [3, 3])
    pkg1 = SlicePackage(base_slice=base, spacing_k=1, size_k=2)
    pkg2 = SlicePackage(base_slice=base, spacing_k=1, size_k=3)

    sset = SlicePackageSet.from_slice_packages([pkg1, pkg2])
    assert sset.package_names == ["package_1", "package_2"]
    assert len(sset.sample_points) == 9 * 2 + 9 * 3

    # a single SlicePackage, not wrapped in a list
    sset_single = SlicePackageSet.from_slice_packages(pkg1)
    assert sset_single.package_names == ["package_1"]
    assert len(sset_single.sample_points) == 9 * 2

    # a single bare SliceGeometry, auto-wrapped as size_k = 1
    sset_slice = SlicePackageSet.from_slice_packages(base)
    assert sset_slice.packages["package_1"].size_k == 1

    # a mixed list of SlicePackage and bare SliceGeometry
    sset_mixed = SlicePackageSet.from_slice_packages([pkg1, base])
    assert sset_mixed.package_names == ["package_1", "package_2"]
    assert sset_mixed.packages["package_2"].size_k == 1
