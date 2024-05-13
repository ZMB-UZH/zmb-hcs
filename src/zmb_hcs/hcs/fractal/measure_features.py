import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import anndata as ad
import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import zarr
from fractal_tasks_core.channels import (
    ChannelInputModel,
    OmeroChannel,
    get_channel_from_image_zarr,
    get_omero_channel_list,
)
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.tables import write_table
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, regionprops_table
from skimage.measure._regionprops import COL_DTYPES

logger = logging.getLogger(__name__)


def most_frequent_value(mask, img):
    masked_img = img[mask].astype(int)
    return np.bincount(masked_img).argmax()


def intensity_std(mask, img):
    masked_img = img[mask]
    return np.std(masked_img)


def intensity_total(mask, img):
    masked_img = img[mask]
    return np.sum(masked_img)


FUNS_PLUS = {
    "most_frequent_value": most_frequent_value,
    "intensity_std": intensity_std,
    "intensity_total": intensity_total,
}


def regionprops_plus(
    label_image,
    intensity_image=None,
    cache=True,
    *,
    extra_properties=None,
    spacing=None,
    offset=None,
):
    """
    Wrapper around regionprops, to integrate extra properties.
    The additional properties are:
    - most_frequent_value
        returns most frequent value of img inside mask
        (mainly used for annotating labels, where img are annotation labels)
    - intensity_std: float
        standard deviation of pixel values
    - intensity_total:
        sum of pixel values
    """
    properties_plus = [most_frequent_value, intensity_std, intensity_total]
    if extra_properties is not None:
        extra_properties = extra_properties + properties_plus
    else:
        extra_properties = properties_plus
    return regionprops(
        label_image=label_image,
        intensity_image=intensity_image,
        cache=cache,
        extra_properties=extra_properties,
        spacing=spacing,
        offset=offset,
    )


def regionprops_table_plus(
    label_image,
    intensity_image=None,
    properties=("label", "bbox"),
    *,
    cache=True,
    separator="-",
    extra_properties=None,
    spacing=None,
):
    """
    Like skimage.measure.regionprops_table(), but incorporates some additional properties:

    - most_frequent_value
        returns most frequent value of img inside mask
        (mainly used for annotating labels, where img are annotation labels)
    - intensity_std: float
        standard deviation of pixel values
    - intensity_total:
        sum of pixel values
    """
    properties_org = []
    properties_plus = []
    for prop in properties:
        if prop in COL_DTYPES.keys():
            properties_org.append(prop)
        elif prop in FUNS_PLUS.keys():
            properties_plus.append(FUNS_PLUS[prop])
    rpt = regionprops_table(
        label_image,
        intensity_image,
        properties=properties_org,
        cache=cache,
        separator=separator,
        extra_properties=properties_plus,
        spacing=spacing,
    )
    return {
        prop: rpt[prop] for prop in properties
    }  # sort table according to input properties


def measure_features_ROI(
    labels,
    annotations_list,
    intensities_list,
    shortest_distance_list,
    ann_prefix_list=None,
    int_prefix_list=None,
    dist_prefix_list=None,
    structure_props=None,
    intensity_props=None,
    pxl_sizes=None,
    optional_columns: dict[str:Any] = {},
):
    """
    Returns measurements of labels

    Args:
        labels: Label image to be measured
        annotations_list: list of label images, which are used to annotate labels
        intensities_list: list of intensity images to measure
        shortest_distance_list: list of label images, for which the shortest distance is calculated
        ann_prefix_list: prefix to use for annotations (default: ann0, ann1, ann2,...)
        int_prefix_list: prefix to use for intensity measurements (default: c0, c1, c2, ...)
        dist_prefix_list: prefix to use for shortest_distance measurements (default: dist0, dist1, dist2, ...)
        structure_props: list of structure properties to measure
        intensity_props: list of intensity properties to measure
        pxl_sizes: list of pixel sizes, must have same length as passed image dimensions
        optional_columns: list of any additional columns and their entries (e.g. {'well':'C01'})
    Returns:
        Pandas dataframe
    """
    # initiate dataframe
    df = pd.DataFrame(index=np.unique(labels)[np.unique(labels) != 0])
    df.index.name = "label"

    # assign labels to annotations
    if ann_prefix_list is None:
        ann_prefix_list = [f"ann{i}" for i in range(len(annotations_list))]
    df_ann_list = []
    for annotations, ann_prefix in zip(annotations_list, ann_prefix_list):
        df_ann = pd.DataFrame(
            regionprops_table_plus(
                labels,
                annotations,
                properties=(
                    [
                        "label",
                        "most_frequent_value",
                    ]
                ),
            )
        )
        df_ann = df_ann.rename(
            columns={
                "most_frequent_value": f"{ann_prefix}_ID",
            }
        )
        df_ann.set_index("label", inplace=True)
        df_ann_list.append(df_ann)

    # do structure measurements
    if structure_props is None:
        structure_props = ["num_pixels"]
    df_struct = pd.DataFrame(
        regionprops_table_plus(
            labels,
            None,
            properties=(
                [
                    "label",
                ]
                + structure_props
            ),
            spacing=pxl_sizes,
        )
    )
    df_struct.set_index("label", inplace=True)

    # do intensity measurements
    if int_prefix_list is None:
        int_prefix_list = [f"c{i}" for i in range(len(intensities_list))]
    if intensity_props is None:
        intensity_props = ["intensity_mean", "intensity_std", "intensity_total"]
    df_int_list = []
    for intensities, int_prefix in zip(intensities_list, int_prefix_list):
        df_int = pd.DataFrame(
            regionprops_table_plus(
                labels,
                intensities,
                properties=(
                    [
                        "label",
                    ]
                    + intensity_props
                ),
                spacing=pxl_sizes,
            )
        )
        df_int = df_int.rename(
            columns={prop: f"{int_prefix}_{prop}" for prop in intensity_props}
        )
        df_int.set_index("label", inplace=True)
        df_int_list.append(df_int)

    # calculated shortest distances
    if dist_prefix_list is None:
        dist_prefix_list = [f"dist{i}" for i in range(len(shortest_distance_list))]
    df_dist_list = []
    for dist_label, dist_prefix in zip(shortest_distance_list, dist_prefix_list):
        dist_transform = distance_transform_edt(
            np.logical_not(dist_label), sampling=pxl_sizes
        )
        df_dist = pd.DataFrame(
            regionprops_table(
                labels,
                dist_transform,
                properties=(
                    [
                        "label",
                        "intensity_min",
                    ]
                ),
                spacing=pxl_sizes,
            )
        )
        df_dist = df_dist.rename(
            columns={
                "intensity_min": f"shortest_distance_to_{dist_prefix}",
            }
        )
        df_dist.set_index("label", inplace=True)
        df_dist_list.append(df_dist)
    # combine all
    df = pd.concat(
        [
            df,
        ]
        + df_ann_list
        + [
            df_struct,
        ]
        + df_int_list
        + df_dist_list,
        axis=1,
    )
    # add additional columns:
    for i, (col_name, col_val) in enumerate(optional_columns.items()):
        df.insert(i, col_name, col_val)
    return df


def measure_features(
    *,
    # Fractal parameters
    zarr_url: str,
    # Task-specific arguments:
    output_table_name: str,
    label_name: str,
    annotation_label_names: Optional[Sequence[str]] = None,
    shortest_distance_label_names: Optional[Sequence[str]] = None,
    channels_to_include: Optional[Sequence[ChannelInputModel]] = None,
    channels_to_exclude: Optional[Sequence[ChannelInputModel]] = None,
    structure_props: Optional[Sequence[str]] = None,
    intensity_props: Optional[Sequence[str]] = None,
    level: int = 0,
    overwrite: bool = True,
) -> None:
    """
    Calculate features based on label image and intensity image (optional).

    Takes a label image and an optional intensity image and calculates
    morphology, intensity and texture features in 2D.

    TODO: Currently, the label image, annotation_images and intensity_images
    all need to have the same resolution. -> Think about how to fix this.

    TODO: Currently only works for well-plates, since it also writes well_name.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).

        output_table_name: Name of the output table.
        label_name: Name of the label that contains the seeds.
            Needs to exist in OME-Zarr file.
        annotation_label_names: List of label names that should be used
            to annotate the labels. Need to exist in OME-Zarr file.
        shortest_distance_label_names: List of label names that should be used
            to calculate the shortest distance to the labels.
            Need to exist in OME-Zarr file.
        channels_to_include: List of channels to include for intensity
            and texture measurements. Use the channel label to indicate
            single channels. If None, all channels are included.
        channels_to_exclude: List of channels to exclude for intensity
            and texture measurements. Use the channel label to indicate
            single channels. If None, no channels are excluded.
        structure_props: List of regionprops structure properties to measure.
        intensity_props: List of regionprops intensity properties to measure.
                ROI_table_name: Name of the ROI table to process.
        level: Resolution of the label image to calculate features.
            Only tested for level 0.
        overwrite: If True, overwrite existing feature table.
    """

    plate_name = Path(zarr_url.split(".zarr/")[0]).name
    component = zarr_url.split(".zarr/")[1]
    well_name = component.split("/")[0] + component.split("/")[1]

    logger.info(f"Calculating {output_table_name} for well {well_name}")

    # get some meta data
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    coarsening_xy = ngff_image_meta.coarsening_xy
    # calculate level pixel sizes:
    level_pxl_sizes_zyx = [
        full_res_pxl_sizes_zyx[0],
    ] + [dn * (coarsening_xy**level) for dn in full_res_pxl_sizes_zyx[1:]]

    # load ROI table
    ROI_table = ad.read_zarr(Path(zarr_url).joinpath("tables", "FOV_ROI_table"))

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    num_ROIs = len(list_indices)

    logger.info(f"Now constructing feature_measurements for {num_ROIs} ROIs")

    measurements_delayed = []
    for i_ROI, indices in enumerate(list_indices):
        # Define region
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (
            slice(0, None),
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )

        # load label image
        label_image_da = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")[
            region[1:]
        ]

        # load annotation images
        annotation_labels_da = []
        if annotation_label_names:
            for annotation_label_name in annotation_label_names:
                annotation_labels_da.append(
                    da.from_zarr(f"{zarr_url}/labels/{annotation_label_name}/{level}")[
                        region[1:]
                    ]
                )

        # load shortest distance images
        shortest_distance_labels_da = []
        if shortest_distance_label_names:
            for shortest_distance_label_name in shortest_distance_label_names:
                shortest_distance_labels_da.append(
                    da.from_zarr(
                        f"{zarr_url}/labels/{shortest_distance_label_name}/{level}"
                    )[region[1:]]
                )

        # load intensity images
        # get all channels in the acquisition and find the ones of interest
        channels = get_omero_channel_list(image_zarr_path=zarr_url)
        if channels_to_include:
            channel_labels_to_include = [c.label for c in channels_to_include]
            channel_wavelength_ids_to_include = [
                c.wavelength_id for c in channels_to_include
            ]
            channels = [
                c
                for c in channels
                if (c.label in channel_labels_to_include)
                or (c.wavelength_id in channel_wavelength_ids_to_include)
            ]
        if channels_to_exclude:
            channel_labels_to_exclude = [c.label for c in channels_to_exclude]
            channel_wavelength_ids_to_exclude = [
                c.wavelength_id for c in channels_to_exclude
            ]
            channels = [
                c
                for c in channels
                if (c.label not in channel_labels_to_exclude)
                and (c.wavelength_id not in channel_wavelength_ids_to_exclude)
            ]
        # loop over channels and load images
        intensity_images_da = []
        for channel in channels:
            tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=channel.wavelength_id,
                label=channel.label,
            )
            ind_channel = tmp_channel.index
            data_zyx = da.from_zarr(f"{zarr_url}/{level}")[region][ind_channel]
            intensity_images_da.append(data_zyx)

        measurement_delayed = dask.delayed(measure_features_ROI)(
            labels=label_image_da,
            annotations_list=annotation_labels_da,
            intensities_list=intensity_images_da,
            shortest_distance_list=shortest_distance_labels_da,
            ann_prefix_list=annotation_label_names,
            int_prefix_list=[channel.wavelength_id for channel in channels],
            dist_prefix_list=shortest_distance_label_names,
            structure_props=structure_props,
            intensity_props=intensity_props,
            pxl_sizes=level_pxl_sizes_zyx,
            optional_columns={
                "plate": plate_name,
                "well": well_name,
                "ROI": ROI_table.obs.index[i_ROI],
            },
        )
        measurements_delayed.append(measurement_delayed)

    logger.info("Now calculating one ROI for meta")

    # get structure of df by calculating first FOV
    # TODO: do this more efficiently
    meta = measurements_delayed[0].compute()

    logger.info("Now calculating features for all ROIs")

    df_measurements = dd.from_delayed(measurements_delayed, meta=meta).compute()

    obs_cols = [
        "label",
        "plate",
        "well",
        "ROI",
    ] + [column for column in df_measurements.columns if "_ID" in column]
    obs = df_measurements.reset_index()[obs_cols]
    obs.index = obs.index.astype(str)
    X = (
        df_measurements.reset_index()
        .drop(columns=obs_cols, errors="ignore")
        .astype("float32")
    )
    X.index = X.index.astype(str)

    feature_table = ad.AnnData(
        X=X,
        obs=obs,
    )

    logger.info("Now writing feature-table")

    # Write to zarr group
    image_group = zarr.group(zarr_url)
    write_table(
        image_group,
        output_table_name,
        feature_table,
        overwrite=overwrite,
        table_attrs={
            "type": "feature_table",
            "region": {"path": f"../labels/{label_name}"},
            "instance_key": "label",
        },
    )

    logger.info("Done.")
