import numpy as np
import SimpleITK as sitk

from ggslicer import ReadImage_fix, SliceGeometry, slice_grid, transform_points


def test_a_real_ants_warp_visibly_distorts_a_grid_built_on_the_fixed_image(testdata_dir):
    base = testdata_dir / "mouse_4"
    fixed = ReadImage_fix(str(base / "fmri_template_on_ccfv3_200um.nii.gz"))
    size = fixed.GetSize()
    center_idx = (size[0] // 2, size[1] // 2, size[2] // 2)
    center_world = fixed.TransformIndexToPhysicalPoint(center_idx)

    geom = SliceGeometry.from_image_axis(fixed, "axial", center_world[2])
    grid_df = slice_grid(geometry=geom)

    # Capture what's needed from the warp image before constructing the
    # DisplacementFieldTransform -- doing so moves/invalidates the source
    # Image object (confirmed directly; see CLAUDE.md), though nothing here
    # needs it afterward anyway.
    inv_warp_img = sitk.ReadImage(str(base / "fmri_template_to_ccfv31InverseWarp.nii.gz"), sitk.sitkVectorFloat64)
    inv_transform = sitk.DisplacementFieldTransform(inv_warp_img)

    warped = transform_points(grid_df, inv_transform)

    # A "grid" line with grid_axis == "i" has, by construction, zero variance
    # in its projection onto direction_i before warping (that's what "fixed
    # local i" means). After a genuine nonlinear warp, that projection should
    # no longer be constant for at least several lines (edge lines
    # near/outside the brain may legitimately see near-zero local
    # deformation).
    origin = geom.origin
    di = geom.direction_i

    is_grid_i = (grid_df["part"] == "grid") & (grid_df["grid_axis"] == "i")
    grid_i = grid_df[is_grid_i]
    warped_i = warped[is_grid_i]  # transform_points() preserves row order/length exactly

    max_std = 0.0
    for line_id in grid_i["line_id"].unique():
        sel = grid_i["line_id"] == line_id
        pts = warped_i[sel][["x", "y", "z"]].to_numpy() - origin
        proj = pts @ di
        max_std = max(max_std, float(np.std(proj)))

    assert max_std > 0.01
