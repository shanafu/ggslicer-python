# ggslicer

This repository is for the `ggslicer` Python package.

Scientific visualization is an integral aspect to sharing research data. In Python, a Grammar of Graphics approach is available through the `plotnine` package. `ggslicer` turns slices of medical and scientific images into tidy `pandas` DataFrames that `plotnine` can plot.

Load a DICOM, NIfTI, MINC, or Analyze image, pick a slice (axis-aligned or oblique), and get back a DataFrame in world coordinates. Segmentations, atlases, contours, grids, and warp fields all land in the same coordinate space, so they overlay directly in one plot.

Built on the `SimpleITK` image processing framework, `ggslicer` handles 2D, 3D, and higher-dimensional (4D/5D) images.

- To find the tutorial, visit: [shanafu/ggslicer-tutorial](https://github.com/shanafu/ggslicer-tutorial)
- To find the equivalent R package, visit: [shanafu/ggslicer](https://github.com/shanafu/ggslicer)

<p>
<img width="1344" height="960" alt="Thalamus atlas overlay on a template brain with labeled structures" src="https://github.com/user-attachments/assets/3edfdf7d-1847-4e44-9916-e48b6472276e" />
<em> An example of a neuroimaging figure created entirely using Grammar of Graphics-based code (via <code>plotnine</code>): an atlas overlay on a template with a specific focus on the thalamus and its labeled structures. </em>
</p>


## Installation

`ggslicer` depends on `SimpleITK`, `numpy`, `pandas`, `plotnine`, `contourpy`, `scipy`, and `tqdm`.To install the package:

```bash
pip install git+https://github.com/shanafu/ggslicer-python.git
```

Check that everything loaded:

```python
import SimpleITK as sitk
import ggslicer
```


## Function reference

Every function below is exported. Use `help(function_name)` for full documentation.


### Slicing and sampling

| Function | Description |
|---|---|
| `slice_image()` | <ul><li>**The main function.**</li><li> Samples an image (object or file path) along an axis/coordinate or a pre-built geometry. </li><li> Returns a DataFrame with the image in `value` and any `extra_images` (e.g. `{"mask": "mask.nii.gz"}`) as additional columns. </li><li> Use `interpolator_overrides` to change the default interpolation for any image. </li></ul>|
| `sample_images()` | <ul><li> Samples one or more images onto an existing `SliceGeometry`, `SlicePackage`, or `SlicePackageSet`. </li><li> Supports 4D/5D images through `extra_index`. </li></ul> |
| `build_slice_geometry()` | <ul><li> Builds a `SlicePackageSet` from an image, an out-of-plane `axis` (`1`/`2`/`3`, `"x"`/`"y"`/`"z"`, or `"sagittal"`/`"coronal"`/`"axial"`), and world `coordinate` values. </li></ul> |
| `discrete_data_names()` | <ul><li> Returns the list of names treated as discrete data (e.g. `"mask"`, `"label"`, `"atlas"`, `"segmentation"`). </li><li> Names are matched by whole token, case-insensitively, so `"brain_mask"` matches. Matching images are always sampled with nearest-neighbour interpolation. </li></ul> |


### Contours

| Function | Description |
|---|---|
| `slice_contours()` | <ul><li> Intensity contour lines of a continuous image at given `levels`, as vertices ready for `geom_path()`. </li></ul>|
| `slice_label_contours()` | <ul><li> Region outlines of a discrete label image (masks, atlases, segmentations). </li></ul> |
| `suggest_contour_levels()` | <ul><li> Suggests `levels` from an image, file path, or `slice_image()` DataFrame, by quantile (default) or by density troughs between populations. Not for discrete data. </li></ul> |


### Label annotations

| Function | Description |
|---|---|
| `contour_centroids()` | <ul><li> One label position per connected region (area-weighted centroid) from `slice_label_contours()` output. </li><li> Supports `label_names` (a dict) and `min_area`. </li></ul> |
| `slice_label_annotations()` | <ul><li> Wrapper that runs `slice_label_contours()` then `contour_centroids()`. </li><li>Use it when you only need the text labels. </li></ul> |
| `slice_label_annotations_layer()` | <ul><li> Returns a ready-made `geom_text()` layer for the output of `slice_label_annotations()`. </li><li>Style with `color`, `size`, `alpha`, `fontweight`. </li></ul>|


### Grid overlay

| Function | Description |
|---|---|
| `slice_grid()` | <ul><li> Bounding box and gridlines for a slice, as line vertices. </li><li> Control density with `spacing`, `point_spacing`, and `padding`. </li></ul> |
| `slice_grid_layers()` | <ul><li> Returns a dict of two styled `geom_path()` layers: `"box"` and `"grid"`. </li></ul>|


### Transforms and warp field arrows

| Function | Description |
|---|---|
| `read_minc_transform()` | <ul><li>Reads an MNI `.xfm` transform file, including `Grid_Transform` and multi-block files, which `SimpleITK.ReadTransform()` can silently mishandle.</li><li> With `corrected=True` (default) the transform is adjusted to work on points from `ReadImage_fix()`-corrected images. </li><li>Set `corrected=False` for the transform exactly as written in the file. </li></ul>|
| `transform_points()` | <ul><li> Applies a SimpleITK transform (object or file path) to the `x`/`y`/`z` columns of any tidy DataFrame. </li><li>`invert=True` applies the inverse, which works for affine transforms only.</li><li> Progress is reported via `tqdm`. </li></ul>|
| `slice_warp_arrows()` | <ul><li> Displacement arrows from a transform, a 3-component displacement-field image, or a file path to either, on a lattice of points in a slice. </li><li> Use `arrow_length` to rescale all arrows to equal length. </li></ul> |
| `slice_warp_arrows_layer()` | <ul><li>Returns a ready-made `geom_segment()` layer with an arrowhead for the output of `slice_warp_arrows()`. </li><li>Style with `color`, `size`, `alpha`, `arrow_length_inches`, `arrow_type`. </li></ul>|


### I/O

| Function | Description |
|---|---|
| `ReadImage_fix()` | <ul><li> Reads an image with SimpleITK; corrects MINC header orientation so it matches an equivalent NIfTI. </li></ul> |
| `WriteImage_fix()` | <ul><li> Writes an image; undoes the MINC correction when writing back to MINC. </li></ul>|
| `orientation_correction()` | <ul><li> The header correction used by the two functions above. It is its own inverse. You usually don't call it directly. </li></ul>|


### Classes

You usually don't need these unless you want to visualize oblique slices and custom stacks. A bare `SliceGeometry` can be used anywhere a `SlicePackage` is expected.

| Class | Description |
|---|---|
| `SliceGeometry` | <ul><li> A single evenly sampled 2D slice in 3D world space (no intensity data). </li><li> Constructors: `SliceGeometry(...)`, `.from_corners()`, `.from_center()`, `.from_normal()`, `.from_bounds()`, `.from_image_axis()`.  </li></ul>|
| `SlicePackage` | <ul><li> A stack of parallel slices along a shared normal. </li><li> Constructors: `SlicePackage(...)`, `.from_extent_k()`, `.from_center_k()`, `.from_slices()`, `.from_image_axis()`. </li></ul>|
| `SlicePackageSet` | <ul><li> A named collection of packages (e.g. several orientations) that produces the final long-format table. </li><li> Constructors: `SlicePackageSet(...)`, `.from_orthogonal_triplet()`, `.from_slice_packages()`. </li></ul>|
