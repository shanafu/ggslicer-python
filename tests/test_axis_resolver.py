import pytest

from ggslicer.geometry import _resolve_axis_index


def test_accepts_numeric_cartesian_and_anatomical_forms():
    for a in (1, "1", "x", "X", "sagittal", "Sagittal"):
        assert _resolve_axis_index(a) == 0
    for a in (2, "2", "y", "coronal"):
        assert _resolve_axis_index(a) == 1
    for a in (3, "3", "z", "axial", "horizontal", "Horizontal"):
        assert _resolve_axis_index(a) == 2


def test_rejects_invalid_input():
    with pytest.raises(ValueError, match="1, 2, or 3"):
        _resolve_axis_index(0)
    with pytest.raises(ValueError, match="1, 2, or 3"):
        _resolve_axis_index(4)
    with pytest.raises(ValueError, match="Invalid `axis`"):
        _resolve_axis_index("bogus")
    with pytest.raises(ValueError, match="must be a single value"):
        _resolve_axis_index(True)
    with pytest.raises(ValueError, match="must be a single value"):
        _resolve_axis_index([1, 2])
