import random
import warnings
from pathlib import Path
from typing import Sequence

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_ROI_table_to_indices,
)


def get_percentile(dask_array, percentiles, bin_width=1):
    # check if int or float & calculate bin_edges
    if np.issubdtype(dask_array.dtype, np.integer):
        mn = np.iinfo(dask_array.dtype).min
        mx = np.iinfo(dask_array.dtype).max
        step = round(bin_width)
        bin_edges = np.arange(mn, mx + step, step)
    elif np.issubdtype(dask_array.dtype, np.float):
        mn = np.finfo(dask_array.dtype).min
        mx = np.finfo(dask_array.dtype).max
        step = bin_width
        bin_edges = np.arange(mn, mx + step, step)
    else:
        raise TypeError("dtype is neither int nor float")
    hist_da, _ = da.histogram(dask_array, bins=bin_edges)
    cumulative_hist = np.cumsum(hist_da.compute())
    total_points = cumulative_hist[-1]
    percentile_indices = np.searchsorted(
        cumulative_hist, np.array(percentiles) * 0.01 * total_points, side="right"
    )
    percentile_values = bin_edges[percentile_indices]
    return percentile_values


def check_same_ROI_shape(list_indices):
    """
    check if all ROIs have the same shape

    Args:
        list_indices: list of ROI indices (generated with convert_ROI_table_to_indices())
    """
    list_indices = np.array(list_indices)
    ref_shape = tuple(list_indices[0][1::2] - list_indices[0][0::2])
    for indices in list_indices:
        shape = tuple(indices[1::2] - indices[0::2])
        if shape != ref_shape:
            raise RuntimeError("ROIs have different shape")
    return ref_shape


def calculate_percentiles(
    *,
    # Standard arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    level: int = 0,
    percentiles: Sequence[float] = (1, 99),
    overwrite_omero: bool = True,
    n_images: int = None,
    random_seed: int = None,
) -> list[list]:
    """
    Calculates percentiles and writes them to omero-channels

    Args:
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Resolution level used to calculate percentiles.
        percentiles: lower and upper percentiles to calculate
        overwrite_omero: if False it will not write into the omero-metadata, and simply
            return the calculated percentile-values
        n_images: randomly pick n_images to calculate percentiles (across fields and planes).
            If None: use all images.
        random_seed: random seed for picking n_images
    """
    # Preliminary checks
    if len(input_paths) > 1:
        raise NotImplementedError
    for percentile in percentiles:
        if not (0 <= percentile <= 100):
            raise RuntimeError("percentiles need to be between 0 and 100")
    if overwrite_omero:
        if len(percentiles) != 2:
            raise RuntimeError("percentiles needs to be of lenth 2")
        if percentiles[0] > percentiles[1]:
            raise RuntimeError("percentiles[0] must be smaller than percentiles[1]")

    # Define zarrurl
    plate, well = component.split(".zarr/")
    in_path = Path(input_paths[0])
    zarrurl = in_path / component

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarrurl)
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)

    # Read FOV ROIs
    FOV_ROI_table = ad.read_zarr(f"{zarrurl}/tables/FOV_ROI_table")

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, "FOV_ROI_table")
    # TODO: implement for differing ROI shapes
    # Note: might break for higher levels
    roi_shape = check_same_ROI_shape(list_indices)

    # lazily load data
    data_czyx = da.from_zarr(f"{zarrurl}/{level}")
    new_shape = (
        len(list_indices),
        data_czyx.shape[0],
    ) + roi_shape
    data_fczyx = da.empty_like(data_czyx, shape=new_shape)
    for i_ROI, indices in enumerate(list_indices):
        # Define region
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (
            slice(0, data_czyx.shape[0]),
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )
        data_fczyx[i_ROI] = data_czyx[region]
    nf, nc, nz, ny, nx = data_fczyx.shape

    # calculate percentiles
    if n_images is None:
        percentile_values_list = [
            get_percentile(data_fczyx[:, c], percentiles) for c in range(nc)
        ]
    elif n_images > (nf * nz):
        warnings.warn(
            "n_images is larger than #FOVs x #planes. "
            f"Using all {nf}x{nz}={nf * nz} images."
        )
        percentile_values_list = [
            get_percentile(data_fczyx[:, c], percentiles) for c in range(nc)
        ]
    else:
        random.seed(random_seed)
        indices = [(f, z) for f in range(nf) for z in range(nz)]
        indices_sample = random.sample(indices, n_images)
        percentile_values_list = []
        for c in range(nc):
            data_sample = np.concatenate(
                [data_fczyx[ind[0], c, ind[1]] for ind in indices_sample]
            )
            percentile_values_list.append(get_percentile(data_sample, percentiles))

    # write omero metadata
    if overwrite_omero:
        with zarr.open(zarrurl, mode="a") as zarr_file:
            omero_dict = zarr_file.attrs["omero"]
            for c, percentile_values in enumerate(percentile_values_list):
                omero_dict["channels"][c]["window"]["start"] = percentile_values[0]
                omero_dict["channels"][c]["window"]["end"] = percentile_values[1]
            zarr_file.attrs["omero"] = omero_dict

    return percentile_values_list
