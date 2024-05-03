import logging
from typing import Optional, Sequence

import anndata as ad
import dask.array as da
import fractal_tasks_core
import numpy as np
import pandas as pd
import zarr
from aicssegmentation.core.utils import hole_filling
from fractal_tasks_core.channels import (
    ChannelInputModel,
    ChannelNotFoundError,
    OmeroChannel,
    get_channel_from_image_zarr,
)
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.masked_loading import masked_loading_wrapper
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    array_to_bounding_box_table,
    check_valid_ROI_indices,
    convert_ROI_table_to_indices,
    empty_bounding_box_table,
    find_overlaps_in_ROI_indices,
    get_overlapping_pairs_3D,
    is_ROI_table_valid,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.cellpose_transforms import (
    CellposeCustomNormalizer,
    normalized_img,
)
from fractal_tasks_core.utils import rescale_datasets
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.segmentation import watershed

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


def gaussian_laplace_threshold(
    struct_img: np.ndarray,
    s2_param: Sequence[Sequence[float]],
):
    """
    Spot-segmentation via laplacian of gaussian.

    Args:
        struct_img: image to be segmented.
        s2_param: list of list of parameters for segmentation.
            The first element is the sigma for the laplacian of gaussian, and the
            second element is the threshold for the filtered image:
            [[sigma1, threshold1], [sigma2, threshold2], ...]
            e.g. [[1, 0.04], [1.5, 0.8], [2, 0.15], [4, 0.20]]
    """
    bw = np.zeros(struct_img.shape, dtype=bool)
    for fid in range(len(s2_param)):
        log_sigma = s2_param[fid][0]
        response = -1 * (log_sigma**2) * ndimage.gaussian_laplace(struct_img, log_sigma)
        bw = np.logical_or(bw, response > s2_param[fid][1])
    return bw


def spot_mask_2D(
    x: np.ndarray,
    gaussian_smoothing_sigma: float,
    s2_param: Sequence[Sequence[float]],
    fill_2d: bool = True,
    fill_max_size: float = 1000,
    normalize: CellposeCustomNormalizer = CellposeCustomNormalizer(),
):
    """
    Spot-segmentation via laplacian of gaussian.

    Args:
        x: 2D image to be segmented
        gaussian_smoothing_sigma: sigma for preprocessing gaussian filter
        s2_param: list of list of parameters for segmentation.
            The first element is the sigma for the laplacian of gaussian, and the
            second element is the threshold for the filtered image:
            [[sigma1, threshold1], [sigma2, threshold2], ...]
            e.g. [[1, 0.04], [1.5, 0.8], [2, 0.15], [4, 0.20]]
        fill_2d: If True, holes will be filled
        fill_max_size: maximum hole-size to be filled
        normalize: By default, data is normalized so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel.
            You can turn off the default rescaling. With the "custom" option,
            you can either provide your own rescaling percentiles or fixed
            rescaling upper and lower bound integers.
    """
    if normalize.type == "default":
        x = normalized_img(
            np.reshape(x, (1,) + x.shape),
            lower_p=1.0,
            upper_p=99.0,
        )[0]
    elif normalize.type == "custom":
        x = normalized_img(
            np.reshape(x, (1,) + x.shape),
            lower_p=normalize.lower_percentile,
            upper_p=normalize.upper_percentile,
            lower_bound=normalize.lower_bound,
            upper_bound=normalize.upper_bound,
        )[0]

    if gaussian_smoothing_sigma:
        x = gaussian(x, sigma=gaussian_smoothing_sigma, preserve_range=True)
    mask = gaussian_laplace_threshold(x, s2_param)
    mask = hole_filling(mask, 0, fill_max_size, fill_2d)
    return mask


def separate_watershed(img, mask, sigma):
    """
    Perform instance segmentation of a mask via watershed:
    1. Gaussian filter image with sigma
    2. Find local intensity maxima (inside mask)
    3. Perform seeded watershed along intensity of image, with maximas as
       seed-points, inside of mask
    """
    # TODO: There are inconsistencies with the watershed algorithm, if there is
    # anisotropy in xy and z
    img_processed = gaussian(img, sigma=sigma)
    coords = peak_local_max(
        img_processed, labels=mask, min_distance=sigma, exclude_border=False
    )
    maximas = np.zeros(img.shape, dtype=bool)
    maximas[tuple(coords.T)] = True
    maximas, _ = ndimage.label(maximas)
    labels = watershed(-img_processed, maximas, mask=mask)
    return labels


def segment_ROI(
    x: np.ndarray,
    gaussian_smoothing_sigma: float,
    s2_param: Sequence[Sequence[float]],
    fill_2d: bool = True,
    fill_max_size: float = 1000,
    normalize: CellposeCustomNormalizer = CellposeCustomNormalizer(),
):
    """
    Instance spot-segmentation via laplacian of gaussian and intensity-watershed

    Args:
        x: 4D numpy array. czyx-image to be segmented. c-dimension should only
            have size 1.
        gaussian_smoothing_sigma: sigma for preprocessing gaussian filter
        s2_param: list of list of parameters for segmentation.
            The first element is the sigma for the laplacian of gaussian, and the
            second element is the threshold for the filtered image:
            [[sigma1, threshold1], [sigma2, threshold2], ...]
            e.g. [[1, 0.04], [1.5, 0.8], [2, 0.15], [4, 0.20]]
        fill_2d: If True, holes will be filled
        fill_max_size: maximum hole-size to be filled
        normalize: By default, data is normalized so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel.
            You can turn off the default rescaling. With the "custom" option,
            you can either provide your own rescaling percentiles or fixed
            rescaling upper and lower bound integers.
    """
    x = x[0]
    mask = np.empty_like(x, dtype="uint16")
    for z in range(x.shape[0]):
        mask[z] = spot_mask_2D(
            x[z],
            gaussian_smoothing_sigma,
            s2_param,
            fill_2d,
            fill_max_size,
            normalize=normalize,
        )
    labels = separate_watershed(x, mask, 1)
    return np.reshape(labels, x.shape)


def segment_particles(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    level: int = 0,
    channel: ChannelInputModel,
    input_ROI_table: str = "FOV_ROI_table",
    output_ROI_table: Optional[str] = None,
    output_label_name: Optional[str] = None,
    # Segmentation parameters
    normalize: CellposeCustomNormalizer = CellposeCustomNormalizer(),
    gaussian_smoothing_sigma: float = None,
    s2_param: Sequence[Sequence[float]] = [
        [1, 0.04],
    ],
    fill_2d: bool = True,
    fill_max_size: float = 1000,
    # Overwrite option
    use_masks: bool = True,
    relabeling: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Segment particles

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Pyramid level of the image to be segmented. Choose `0` to
            process at full resolution.
        channel: Primary channel for segmentation; requires either
            `wavelength_id` (e.g. `A01_C01`) or `label` (e.g. `DAPI`).
        input_ROI_table: Name of the ROI table over which the task loops to
            apply segmentation. Examples: `FOV_ROI_table` => loop over
            the field of views, `organoid_ROI_table` => loop over the organoid
            ROI table (generated by another task), `well_ROI_table` => process
            the whole well as one image.
        output_ROI_table: If provided, a ROI table with that name is created,
            which will contain the bounding boxes of the newly segmented
            labels. ROI tables should have `ROI` in their name.
        output_label_name: Name of the output label image (e.g. `"organoids"`).
        normalize: By default, data is normalized so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel.
            This automatic normalization can lead to issues when the image to
            be segmented is very sparse. You can turn off the default
            rescaling. With the "custom" option, you can either provide your
            own rescaling percentiles or fixed rescaling upper and lower
            bound integers.
        gaussian_smoothing_sigma: sigma for preprocessing gaussian filter
            (in pixels @ level0)
        s2_param: list of list of parameters for segmentation.
            The first element is the sigma for the laplacian of gaussian, and the
            second element is the threshold for the filtered image:
            [[sigma1, threshold1], [sigma2, threshold2], ...]
            e.g. [[1, 0.04], [1.5, 0.8], [2, 0.15], [4, 0.20]]
            (sigma in pixels @ level0)
        fill_2d: If True, holes will be filled
        fill_max_size: maximum hole-size to be filled (in pixels @ level0)
        use_masks: If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `organoid_ROI_table`).
        relabeling: If `True`, apply relabeling so that label values are
            unique for all objects in the well.
        overwrite: If `True`, overwrite the task output.

    """

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    actual_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    logger.info(f"NGFF image has {num_levels=}")
    logger.info(f"NGFF image has {coarsening_xy=}")
    logger.info(f"NGFF image has full-res pixel sizes {full_res_pxl_sizes_zyx}")
    logger.info(
        f"NGFF image has level-{level} pixel sizes " f"{actual_res_pxl_sizes_zyx}"
    )

    # Find channel index
    try:
        tmp_channel: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=zarr_url,
            wavelength_id=channel.wavelength_id,
            label=channel.label,
        )
    except ChannelNotFoundError as e:
        logger.warning(
            "Channel not found, exit from the task.\n" f"Original error: {str(e)}"
        )
        return {}
    ind_channel = tmp_channel.index

    # Set channel label
    if output_label_name is None:
        try:
            channel_label = tmp_channel.label
            output_label_name = f"label_{channel_label}"
        except (KeyError, IndexError):
            output_label_name = f"label_{ind_channel}"

    # Load ZYX data
    data_zyx = da.from_zarr(f"{zarr_url}/{level}")[ind_channel]
    logger.info(f"{data_zyx.shape=}")

    # Read ROI table
    ROI_table_path = f"{zarr_url}/tables/{input_ROI_table}"
    ROI_table = ad.read_zarr(ROI_table_path)

    # Perform some checks on the ROI table
    valid_ROI_table = is_ROI_table_valid(table_path=ROI_table_path, use_masks=use_masks)
    if use_masks and not valid_ROI_table:
        logger.info(
            f"ROI table at {ROI_table_path} cannot be used for masked "
            "loading. Set use_masks=False."
        )
        use_masks = False
    logger.info(f"{use_masks=}")

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, input_ROI_table)

    # If we are not planning to use masked loading, fail for overlapping ROIs
    if not use_masks:
        overlap = find_overlaps_in_ROI_indices(list_indices)
        if overlap:
            raise ValueError(
                f"ROI indices created from {input_ROI_table} table have "
                "overlaps, but we are not using masked loading."
            )

    # Rescale datasets (only relevant for level>0)
    if ngff_image_meta.axes_names[0] != "c":
        raise ValueError(
            "Cannot set `remove_channel_axis=True` for multiscale "
            f"metadata with axes={ngff_image_meta.axes_names}. "
            'First axis should have name "c".'
        )
    new_datasets = rescale_datasets(
        datasets=[ds.dict() for ds in ngff_image_meta.datasets],
        coarsening_xy=coarsening_xy,
        reference_level=level,
        remove_channel_axis=True,
    )

    label_attrs = {
        "image-label": {
            "version": __OME_NGFF_VERSION__,
            "source": {"image": "../../"},
        },
        "multiscales": [
            {
                "name": output_label_name,
                "version": __OME_NGFF_VERSION__,
                "axes": [
                    ax.dict()
                    for ax in ngff_image_meta.multiscale.axes
                    if ax.type != "channel"
                ],
                "datasets": new_datasets,
            }
        ],
    }

    image_group = zarr.group(zarr_url)
    label_group = prepare_label_group(
        image_group,
        output_label_name,
        overwrite=overwrite,
        label_attrs=label_attrs,
        logger=logger,
    )

    logger.info(f"Helper function `prepare_label_group` returned {label_group=}")
    logger.info(f"Output label path: {zarr_url}/labels/{output_label_name}/0")
    store = zarr.storage.FSStore(f"{zarr_url}/labels/{output_label_name}/0")
    label_dtype = np.uint32

    # Ensure that all output shapes & chunks are 3D (for 2D data: (1, y, x))
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/398
    shape = data_zyx.shape
    if len(shape) == 2:
        shape = (1, *shape)
    chunks = data_zyx.chunksize
    if len(chunks) == 2:
        chunks = (1, *chunks)
    mask_zarr = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=False,
        dimension_separator="/",
    )

    logger.info(
        f"mask will have shape {data_zyx.shape} " f"and chunks {data_zyx.chunks}"
    )

    # Counters for relabeling
    if relabeling:
        num_labels_tot = 0

    # Iterate over ROIs
    num_ROIs = len(list_indices)

    if output_ROI_table:
        bbox_dataframe_list = []

    logger.info(f"Now starting loop over {num_ROIs} ROIs")
    for i_ROI, indices in enumerate(list_indices):
        # Define region
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )
        logger.info(f"Now processing ROI {i_ROI+1}/{num_ROIs}")

        img_np = np.expand_dims(
            load_region(data_zyx, region, compute=True, return_as_3D=True),
            axis=0,
        )

        # Prepare keyword arguments for segment_ROI function
        kwargs_segment_ROI = dict(
            gaussian_smoothing_sigma=gaussian_smoothing_sigma,
            s2_param=s2_param,
            fill_2d=fill_2d,
            fill_max_size=fill_max_size,
            normalize=normalize,
        )

        # Prepare keyword arguments for preprocessing function
        preprocessing_kwargs = {}
        if use_masks:
            preprocessing_kwargs = dict(
                region=region,
                current_label_path=f"{zarr_url}/labels/{output_label_name}/0",
                ROI_table_path=ROI_table_path,
                ROI_positional_index=i_ROI,
            )

        # Call segment_ROI through the masked-loading wrapper, which includes
        # pre/post-processing functions if needed
        new_label_img = masked_loading_wrapper(
            image_array=img_np,
            function=segment_ROI,
            kwargs=kwargs_segment_ROI,
            use_masks=use_masks,
            preprocessing_kwargs=preprocessing_kwargs,
        )

        # Shift labels and update relabeling counters
        if relabeling:
            num_labels_roi = np.max(new_label_img)
            new_label_img[new_label_img > 0] += num_labels_tot
            num_labels_tot += num_labels_roi

            # Write some logs
            logger.info(f"ROI {indices}, {num_labels_roi=}, {num_labels_tot=}")

            # Check that total number of labels is under control
            if num_labels_tot > np.iinfo(label_dtype).max:
                raise ValueError(
                    "ERROR in re-labeling:"
                    f"Reached {num_labels_tot} labels, "
                    f"but dtype={label_dtype}"
                )

        if output_ROI_table:
            bbox_df = array_to_bounding_box_table(
                new_label_img,
                actual_res_pxl_sizes_zyx,
                origin_zyx=(s_z, s_y, s_x),
            )

            bbox_dataframe_list.append(bbox_df)

            overlap_list = []
            for df in bbox_dataframe_list:
                overlap_list.extend(
                    get_overlapping_pairs_3D(df, full_res_pxl_sizes_zyx)
                )
            if len(overlap_list) > 0:
                logger.warning(f"{len(overlap_list)} bounding-box pairs overlap")

        # Compute and store 0-th level to disk
        da.array(new_label_img).to_zarr(
            url=mask_zarr,
            region=region,
            compute=True,
        )

    logger.info(f"End segmentation task for {zarr_url}, " "now building pyramids.")

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{output_label_name}",
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info("End building pyramids")

    if output_ROI_table:
        # Handle the case where `bbox_dataframe_list` is empty (typically
        # because list_indices is also empty)
        if len(bbox_dataframe_list) == 0:
            bbox_dataframe_list = [empty_bounding_box_table()]
        # Concatenate all ROI dataframes
        df_well = pd.concat(bbox_dataframe_list, axis=0, ignore_index=True)
        df_well.index = df_well.index.astype(str)
        # Extract labels and drop them from df_well
        labels = pd.DataFrame(df_well["label"].astype(str))
        df_well.drop(labels=["label"], axis=1, inplace=True)
        # Convert all to float (warning: some would be int, in principle)
        bbox_dtype = np.float32
        df_well = df_well.astype(bbox_dtype)
        # Convert to anndata
        bbox_table = ad.AnnData(df_well, dtype=bbox_dtype)
        bbox_table.obs = labels

        # Write to zarr group
        image_group = zarr.group(zarr_url)
        logger.info(
            "Now writing bounding-box ROI table to "
            f"{zarr_url}/tables/{output_ROI_table}"
        )
        table_attrs = {
            "type": "masking_roi_table",
            "region": {"path": f"../labels/{output_label_name}"},
            "instance_key": "label",
        }
        write_table(
            image_group,
            output_ROI_table,
            bbox_table,
            overwrite=overwrite,
            table_attrs=table_attrs,
        )
