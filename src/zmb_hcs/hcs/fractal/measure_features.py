from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import logging
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
    ann_prefix_list=None,
    int_prefix_list=None,
    structure_props=None,
    intensity_props=None,
    optional_columns: dict[str:Any] = {},
):
    """
    Returns measurements of labels

    Args:
        labels: Label image to be measured
        annotations_list: list of label images, which are used to annotate labels
        intensities_list: list of intensity images to measure
        ann_prefix_list: prefix to use for annotations (default: ann0, ann1, ann2,...)
        int_prefix_list: prefix to use for intensity measurements (default: c0, c1, c2, ...)
        structure_props: list of structure properties to measure
        intensity_props: list of intensity properties to measure
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
            )
        )
        df_int = df_int.rename(
            columns={prop: f"{int_prefix}_{prop}" for prop in intensity_props}
        )
        df_int.set_index("label", inplace=True)
        df_int_list.append(df_int)

    # combine all
    df = pd.concat(
        [
            df,
        ]
        + df_ann_list
        + [
            df_struct,
        ]
        + df_int_list,
        axis=1,
    )
    # add additional columns:
    for i, (col_name, col_val) in enumerate(optional_columns.items()):
        df.insert(i, col_name, col_val)
    return df


def measure_features(
    *,
    # Default arguments for fractal tasks:
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments:
    output_table_name: str,
    label_image_name: str,
    annotation_image_names: Optional[Sequence[str]] = None,
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

    Args:
        input_paths: Path to the parent folder of the NGFF image.
            This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This argument is not used in this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path of the NGFF image, relative to `input_paths[0]`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This argument is not used in this task.
            (standard argument for Fractal tasks, managed by Fractal server).
                output_table_name: Name of the feature table.
        label_image_name: Name of the label image that contains the seeds.
            Needs to exist in OME-Zarr file.
        annotation_image_names: List of the label images that should be used
            to annotate the labels. Need to exist in OME-Zarr file.
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

    plate, well = component.split(".zarr/")
    well_name = component.split("/")[1] + component.split("/")[2]
    in_path = Path(input_paths[0])
    zarrurl = (in_path / component).as_posix()

    logger.info(f"Now processing well {well_name}")

    # get some meta data
    ngff_image_meta = load_NgffImageMeta(in_path.joinpath(component))
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    coarsening_xy = ngff_image_meta.coarsening_xy

    # load ROI table
    ROI_table = ad.read_zarr(in_path.joinpath(component, "tables", "FOV_ROI_table"))

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
        label_image_da = da.from_zarr(f"{zarrurl}/labels/{label_image_name}/{level}")[
            region[1:]
        ]

        # load annotation images
        annotation_images_da = []
        if annotation_image_names:
            for annotation_image_name in annotation_image_names:
                annotation_images_da.append(
                    da.from_zarr(f"{zarrurl}/labels/{annotation_image_name}/{level}")[
                        region[1:]
                    ]
                )

        # load intensity images
        # get all channels in the acquisition and find the ones of interest
        channels = get_omero_channel_list(image_zarr_path=zarrurl)
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
                image_zarr_path=zarrurl,
                wavelength_id=channel.wavelength_id,
                label=channel.label,
            )
            ind_channel = tmp_channel.index
            data_zyx = da.from_zarr(f"{zarrurl}/{level}")[region][ind_channel]
            intensity_images_da.append(data_zyx)

        # measurements = measure_features_ROI(
        #     labels = label_image_da.compute(),
        #     annotations_list = [annotation_image_da.compute() for annotation_image_da in annotation_images_da],
        #     intensities_list = [intensity_image_da.compute() for intensity_image_da in intensity_images_da],
        #     ann_prefix_list=annotation_image_names,
        #     int_prefix_list=[channel.wavelength_id for channel in channels],
        #     structure_props=structure_props,
        #     intensity_props=intensity_props,
        #     optional_columns={'plate':plate, 'well':well_name, 'FOV':i_ROI}
        # )
        measurement_delayed = dask.delayed(measure_features_ROI)(
            labels=label_image_da,
            annotations_list=annotation_images_da,
            intensities_list=intensity_images_da,
            ann_prefix_list=annotation_image_names,
            int_prefix_list=[channel.wavelength_id for channel in channels],
            structure_props=structure_props,
            intensity_props=intensity_props,
            optional_columns={
                "plate": plate,
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
    image_group = zarr.group(zarrurl)
    write_table(
        image_group,
        output_table_name,
        feature_table,
        overwrite=overwrite,
        table_attrs={
            "type": "feature_table",
            "region": {"path": f"../labels/{label_image_name}"},
            "instance_key": "label",
        },
    )

    logger.info("Done.")