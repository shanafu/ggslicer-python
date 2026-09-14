import numpy as np
import SimpleITK as sitk

from ggslicer import ReadImage_fix, SliceGeometry, sample_images, build_slice_geometry, slice_image


def test_slice_image_with_human_1s_annotation_uses_nearest_neighbor_by_name(testdata_dir):
    base = testdata_dir / "human_1"
    template = ReadImage_fix(str(base / "mni_icbm152_t1_tal_nlin_sym_09b_hires.nii"))
    # Cast to float: the raw annotation is uint32, and sitk.Resample() preserves the
    # input pixel type, which would silently round a forced-linear sample back to an
    # integer and defeat the point of this test (checking that linear vs.
    # nearest-neighbor actually differ).
    annotation = sitk.Cast(ReadImage_fix(str(base / "annotation.nii.gz")), sitk.sitkFloat32)

    # build_slice_geometry()/from_image_axis() snaps the in-plane sample grid exactly
    # onto the source image's own voxel grid, so on-grid sampling can never
    # distinguish nearest-neighbor from linear (both land exactly on a voxel center).
    # Shift the in-plane origin by a quarter voxel (via the geometry= escape hatch) so
    # samples genuinely fall between annotation's voxel centers, crossing the many
    # label boundaries confirmed (by direct inspection) to exist at z=8.0.
    on_grid = build_slice_geometry(template, "axial", 8.0)
    base_slice = on_grid.packages["package_1"].base_slice
    quarter_voxel = 0.25 * (
        base_slice.direction_i * base_slice.spacing[0] + base_slice.direction_j * base_slice.spacing[1]
    )
    off_grid = base_slice.translate(quarter_voxel)

    out = slice_image(template, geometry=off_grid, extra_images={"annotation": annotation})
    vals = out["annotation"].dropna().to_numpy()
    assert np.all(np.abs(vals - np.round(vals)) < 1e-6)

    out_linear = slice_image(
        template, geometry=off_grid,
        extra_images={"annotation": annotation},
        interpolator_overrides={"annotation": sitk.sitkLinear},
    )
    vals_linear = out_linear["annotation"].dropna().to_numpy()
    assert np.any(np.abs(vals_linear - np.round(vals_linear)) > 1e-6)


def test_slice_image_works_correctly_on_both_minc_and_nifti_loaded_versions_of_the_same_template(testdata_dir):
    # ReadImage_fix()'s orientation_correction() corrects MINC's header
    # (origin/direction) so it describes the same real-world locations an
    # independently-converted NIfTI file of the same anatomy does --
    # verified directly (mincheader/fslhd ground truth; see CLAUDE.md and
    # test_io.py) that this now matches *exactly*, not just in sign
    # convention. An earlier version of this test/comment assumed a
    # genuine, expected mismatch here ("the brain isn't vertically
    # centered") -- that was actually the bug this fix addresses, not a
    # real anatomical asymmetry.
    base = testdata_dir / "human_1"
    mnc = ReadImage_fix(str(base / "mni_icbm152_t1_tal_nlin_sym_09b_hires.mnc"))
    nii = ReadImage_fix(str(base / "mni_icbm152_t1_tal_nlin_sym_09b_hires.nii"))

    assert all(v > 0 for v in mnc.GetOrigin()[:2])
    assert all(v > 0 for v in nii.GetOrigin()[:2])
    assert np.allclose(mnc.GetOrigin(), nii.GetOrigin())
    assert np.allclose(mnc.GetDirection(), nii.GetDirection())

    out_mnc = slice_image(mnc, axis="axial", coordinate=mnc.GetOrigin()[2])
    out_nii = slice_image(nii, axis="axial", coordinate=nii.GetOrigin()[2])

    assert len(out_mnc) == 394 * 466
    assert len(out_nii) == 394 * 466
    assert not out_mnc["value"].isna().all()
    assert not out_nii["value"].isna().all()

    # since the two now share identical geometry, sampling the same nominal
    # slice should give near-identical intensities.
    valid = out_mnc["value"].notna() & out_nii["value"].notna()
    corr = np.corrcoef(out_mnc.loc[valid, "value"], out_nii.loc[valid, "value"])[0, 1]
    assert corr > 0.999


def test_mouse_1s_mask_column_is_usable_for_tidy_side_filtering(testdata_dir):
    base = testdata_dir / "mouse_1"
    average = ReadImage_fix(str(base / "DSURQE_40micron_average.mnc"))
    mask = ReadImage_fix(str(base / "DSURQE_40micron_mask.mnc"))
    labels = ReadImage_fix(str(base / "DSURQE_40micron_labels.mnc"))

    out = slice_image(average, axis="coronal", coordinate=0, extra_images={"mask": mask, "labels": labels})
    assert {"mask", "labels"}.issubset(out.columns)

    filtered = out.query("mask > 0")
    assert len(filtered) < len(out)


def test_build_slice_geometry_places_human_3s_uneven_slice_coordinates_correctly(testdata_dir):
    base = testdata_dir / "human_3"
    template = ReadImage_fix(str(base / "mni_icbm152_t1_tal_nlin_sym_09b_hires.nii"))

    pset = build_slice_geometry(template, "coronal", [9.0, 8.5])
    assert len(pset.package_names) == 2

    pts = pset.sample_points
    y_by_package = pts.groupby("package")["y"].apply(lambda v: round(float(v.iloc[0]), 4))
    assert set(y_by_package) == {9.0, 8.5}


def test_sample_images_combines_mouse_2s_multi_resolution_template_labels_gene_expression(testdata_dir):
    base = testdata_dir / "mouse_2"
    labels = ReadImage_fix(str(base / "AMBA_relabeled_backsampled_50um.mnc"))
    template25 = ReadImage_fix(str(base / "average_template_25.mnc"))
    bdnf = ReadImage_fix(str(base / "Bdnf_79587720.mnc"))

    geom = SliceGeometry.from_image_axis(labels, "coronal", labels.GetOrigin()[1])
    out = sample_images(geom, {"labels": labels, "template": template25, "bdnf": bdnf})

    assert {"labels", "template", "bdnf"}.issubset(out.columns)
    assert not out["template"].isna().all()
    assert not out["bdnf"].isna().all()
