# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
#
# Adapted by:
# Adrian Tschan <adrian.tschan@uzh.ch>
# Flurin Sturzenegger <st.flurin@gmail.com>
#
# This file is based on Fractal code originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import anndata as ad
import dask.array as da

# from dask.distributed import Client, wait
import numpy as np
import zarr

# from basicpy import BaSiC
# from pydantic.decorator import validate_arguments
from fractal_tasks_core.channels import OmeroChannel, get_omero_channel_list
from fractal_tasks_core.ngff import load_NgffImageMeta

# from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_ROI_table_to_indices,
)
from zmb_hcs.hcs.fractal.pyramids import build_pyramid

logger = logging.getLogger(__name__)


def correct(
    img_stack: np.ndarray,
    flatfield: np.ndarray,
    darkfield: np.ndarray,
    baseline: int,
):
    """
    Apply illumination correction to all fields of view.

    Corrects a stack of images, using a given illumination profile (e.g. bright
    in the center of the image, dim outside).

    Args:
        img_stack: 4D numpy array (czyx), with dummy size along c.
        flatfield: 2D numpy array (yx)
        darkfield: 2D numpy array (yx)
        baseline: baseline value to be subtracted from the image
    """

    logger.info(f"Start correct, {img_stack.shape}")

    # Check shapes
    if flatfield.shape != img_stack.shape[2:] or img_stack.shape[0] != 1:
        raise ValueError(
            "Error in illumination_correction:\n"
            f"{img_stack.shape=}\n{flatfield.shape=}"
        )

    # Store info about dtype
    dtype = img_stack.dtype
    dtype_max = np.iinfo(dtype).max

    #  Apply the normalized correction matrix (requires a float array)
    # img_stack = img_stack.astype(np.float64)
    new_img_stack = (img_stack - darkfield) / flatfield[None, None, :, :]

    # Background subtraction
    new_img_stack = np.where(
        new_img_stack > baseline,
        new_img_stack - baseline,
        0,
    )

    # Handle edge case: corrected image may have values beyond the limit of
    # the encoding, e.g. beyond 65535 for 16bit images. This clips values
    # that surpass this limit and triggers a warning
    if np.sum(new_img_stack > dtype_max) > 0:
        warnings.warn(
            "Illumination correction created values beyond the max range of "
            f"the current image type. These have been clipped to {dtype_max=}."
        )
        new_img_stack[new_img_stack > dtype_max] = dtype_max

    logger.info("End correct")

    # Cast back to original dtype and return
    return new_img_stack.astype(dtype)


# @validate_arguments
def apply_basicpy_illumination_model(
    *,
    # Fractal parameters
    zarr_url: str,
    # Task-specific arguments
    illumination_profiles_folder: str,
    overwrite_input: bool = True,
    new_well_sub_group: Optional[str] = None,
) -> dict[str, Any]:
    """
    Applies illumination correction to the images in the OME-Zarr.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles_folder: Path of folder of illumination profiles.
        overwrite_input:
            If `True`, the results of this task will overwrite the input image
            data. In the current version, `overwrite_input=False` is not
            implemented.
        new_well_sub_group: Name of new well-subgroup. If this is set,
            overwrite_input needs to be False.
            Example: `0_illumination_corrected`.
    """

    # Preliminary checks
    if (overwrite_input and new_well_sub_group is not None) or (
        new_well_sub_group is None and not overwrite_input
    ):
        raise ValueError(f"{overwrite_input=}, but {new_well_sub_group=}")

    if not overwrite_input:
        msg = (
            "We still have to harmonize illumination_correction("
            "overwrite_input=False) with replicate_zarr_structure(..., "
            "suffix=..)"
        )
        raise NotImplementedError(msg)

    # Define old/new zarrurls
    _, well = zarr_url.split(".zarr/")
    zarrurl_old = zarr_url
    if overwrite_input:
        zarrurl_new = zarrurl_old
    else:
        old_well_sub_group = zarrurl_old.split("/")[-1]
        if old_well_sub_group == new_well_sub_group:
            raise ValueError(
                f"{old_well_sub_group=}, which is the same as"
                f" {new_well_sub_group=}, but {overwrite_input=}."
            )
        zarrurl_new = (Path(zarr_url).parent / new_well_sub_group).as_posix()

    t_start = time.perf_counter()
    logger.info("Start illumination_correction")
    logger.info(f"  {overwrite_input=}")
    logger.info(f"  {zarrurl_old=}")
    logger.info(f"  {zarrurl_new=}")

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarrurl_old)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    logger.info(f"NGFF image has {num_levels=}")
    logger.info(f"NGFF image has {coarsening_xy=}")
    logger.info(f"NGFF image has full-res pixel sizes {full_res_pxl_sizes_zyx}")

    # Read channels from .zattrs
    channels: list[OmeroChannel] = get_omero_channel_list(image_zarr_path=zarrurl_old)
    num_channels = len(channels)

    # Read FOV ROIs
    FOV_ROI_table = ad.read_zarr(f"{zarrurl_old}/tables/FOV_ROI_table")

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, "FOV_ROI_table")

    # Extract image size from FOV-ROI indices. Note: this works at level=0,
    # where FOVs should all be of the exact same size (in pixels)
    ref_img_size = None
    for indices in list_indices:
        img_size = (indices[3] - indices[2], indices[5] - indices[4])
        if ref_img_size is None:
            ref_img_size = img_size
        else:
            if img_size != ref_img_size:
                raise ValueError("ERROR: inconsistent image sizes in list_indices")

    # Lazily load highest-res level from original zarr array
    data_czyx = da.from_zarr(f"{zarrurl_old}/0")

    # Create zarr for output
    if overwrite_input:
        fov_path = zarrurl_old
        new_zarr = zarr.open(f"{zarrurl_old}/0")
    else:
        fov_path = zarrurl_new
        new_zarr = zarr.create(
            shape=data_czyx.shape,
            chunks=data_czyx.chunksize,
            dtype=data_czyx.dtype,
            store=zarr.storage.FSStore(f"{zarrurl_new}/0"),
            overwrite=False,
            dimension_separator="/",
        )

    # Iterate over FOV ROIs
    num_ROIs = len(list_indices)
    for i_c, channel in enumerate(channels):
        # load illumination model
        logger.info(f"loading illumination model for channel {channel.label}")
        # basic = BaSiC()
        # basic = basic.load_model(
        #     illumination_profiles_folder + f"/{channel.label}")
        folder_path = Path(illumination_profiles_folder) / f"{channel.label}"
        flatfield = np.load(folder_path / "flatfield.npy")
        darkfield = np.load(folder_path / "darkfield.npy")
        baseline = np.load(folder_path / "baseline.npy")

        for i_ROI, indices in enumerate(list_indices):
            # Define region
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(i_c, i_c + 1),
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            logger.info(
                f"Now processing ROI {i_ROI+1}/{num_ROIs} "
                f"for channel {i_c+1}/{num_channels}"
            )

            # Execute illumination correction
            corrected_fov = correct(
                img_stack=data_czyx[region].compute(),
                flatfield=flatfield,
                darkfield=darkfield,
                baseline=int(np.median(baseline)),
            )
            da.array(corrected_fov).to_zarr(
                url=new_zarr,
                region=region,
                compute=True,
            )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=fov_path,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_czyx.chunksize,
    )

    t_end = time.perf_counter()
    logger.info(f"End illumination_correction, elapsed: {t_end-t_start}")

    return {}


# if __name__ == "__main__":
#     from fractal_tasks_core.tasks._utils import run_fractal_task

#     run_fractal_task(
#         task_function=apply_basicpy_illumination_model,
#         logger_name=logger.name,
#     )
