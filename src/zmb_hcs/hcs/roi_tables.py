# This file was copied from
# https://github.com/jluethi/fractal-faim-hcs/tree/20_align_with_faim_hcs_023
# /src/fractal_faim_hcs/roi_tables.py
# commit 6e7fca5
# (direct installation from there doesn't work because it has faim-hcs
# as a dependency, which gives installation errors for me)

import anndata as ad
import numpy as np
import pandas as pd
from faim_ipa.hcs.acquisition import PlateAcquisition, WellAcquisition
from faim_ipa.stitching import Tile


def create_ROI_tables(plate_acquistion: PlateAcquisition):
    """Generate ROI tables for all images in a plate."""
    columns = [
        "FieldIndex",
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    plate_roi_tables = {}
    for well_acquisition in plate_acquistion.get_well_acquisitions():
        # Get pixel sizes
        xy_spacing = well_acquisition.get_yx_spacing()
        z_spacing = well_acquisition.get_z_spacing()
        # XXX EDITED TO ALSO WORK FOR SINGLE PLANE ACQUISITIONS:
        if z_spacing is None:
            z_spacing = 1
        pixel_size_zyx = (z_spacing, *xy_spacing)

        # Create ROI tables
        plate_roi_tables[well_acquisition.name] = {
            "FOV_ROI_table": create_fov_ROI_table(
                well_acquisition.get_tiles(),
                columns,
                pixel_size_zyx,
            ),
            "well_ROI_table": create_well_ROI_table(
                well_acquisition,
                columns,
                pixel_size_zyx,
            ),
        }

    return plate_roi_tables


def create_well_ROI_table(
    well_acquisition: WellAcquisition,
    columns: list[str],
    pixel_size_zyx: list[float],
):
    """Generate a well ROI table."""
    well_roi = [
        "well_1",
        0.0,
        0.0,
        0.0,
        well_acquisition.get_shape()[-1] * pixel_size_zyx[2],
        well_acquisition.get_shape()[-2] * pixel_size_zyx[1],
        well_acquisition.get_shape()[-3] * pixel_size_zyx[0],
    ]
    well_roi_table = pd.DataFrame(well_roi).T
    well_roi_table.columns = columns
    well_roi_table.set_index("FieldIndex", inplace=True)
    # Cast the values to float to avoid anndata type issues
    well_roi_table = well_roi_table.astype(np.float32)
    return ad.AnnData(well_roi_table)


def create_fov_ROI_table(
    tiles: list[Tile], columns: list[str], pixel_size_zyx: list[float]
):
    """Generate a FOV ROI table based on the position of the tiles."""
    fov_rois = []
    tile = tiles[0]
    min_z = tile.position.z * pixel_size_zyx[0]
    max_z = (tile.position.z + 1) * pixel_size_zyx[0]
    fov_counter = 1
    for tile in tiles:
        z_start = tile.position.z * pixel_size_zyx[0]
        z_end = (tile.position.z + 1) * pixel_size_zyx[0]
        if z_start < min_z:
            min_z = z_start

        if z_end > max_z:
            max_z = z_end

        if tile.position.z == 0 and tile.position.channel == 0:
            fov_rois.append(
                (
                    f"FOV_{fov_counter}",
                    tile.position.x * pixel_size_zyx[2],
                    tile.position.y * pixel_size_zyx[1],
                    tile.position.z * pixel_size_zyx[0],
                    tile.shape[-1] * pixel_size_zyx[2],
                    tile.shape[-2] * pixel_size_zyx[1],
                    (tile.position.z + 1) * pixel_size_zyx[0],
                )
            )
            fov_counter += 1
    roi_table = pd.DataFrame(fov_rois, columns=columns).set_index("FieldIndex")

    roi_table["z_micrometer"] = min_z
    roi_table["len_z_micrometer"] = max_z
    # Cast the values to float to avoid anndata type issues
    roi_table = roi_table.astype(np.float32)
    return ad.AnnData(roi_table)