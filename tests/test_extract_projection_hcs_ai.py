"""Unit tests for extract_projection_hcs_ai (no microscope data required)."""

from zmb_hcs.hcs.extract_projection_hcs_ai import (
    _experiment_holds_projection,
    _projection_filename,
)


def test_projection_filename():
    assert (
        _projection_filename("test_data_t0_C05_s0_w0_z0.tif")
        == "test_data_projection_t0_C05_s0_w0_z0.tif"
    )
    assert (
        _projection_filename("plate_t12_C06_s3_w1_z2.tif")
        == "plate_projection_t12_C06_s3_w1_z2.tif"
    )


def test_experiment_holds_projection():
    # Files with the ``_projection`` marker are real projections.
    proj = {
        "test_data_projection_t0_C05_s0_w0_z0.tif": {},
        "__columns__": ["ImageFileName"],
    }
    assert _experiment_holds_projection(proj) is True

    # Raw single-plane / center-Z acquisitions keep plain ``_z{N}`` names.
    raw = {
        "test_data_t0_C05_s0_w0_z1.tif": {},
        "__columns__": ["ImageFileName"],
    }
    assert _experiment_holds_projection(raw) is False

    # Empty / missing experiment folder.
    assert _experiment_holds_projection({}) is False
