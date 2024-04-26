# Original authors:
# Adrian Tschan <adrian.tschan@uzh.ch>
#
# This file is part of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
#
# Adapted for zmb_hcs by:
# Flurin Sturzenegger <st.flurin@gmail.com>

import os
import shutil
import logging
import random
from pathlib import Path
#from typing import Any
#from typing import Sequence

import anndata as ad
from basicpy import BaSiC
import dask.array as da
import numpy as np
#import pandas as pd
import zarr
#from pydantic.decorator import validate_arguments

from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)

# XXX for basicpy, maybe add:
import jax
jax.config.update("jax_platform_name", "cpu")


logger = logging.getLogger(__name__)


#@validate_arguments
def calculate_illumination_profiles(
    *,
    # Standard arguments
    input_zarr: Path,
    # Task-specific arguments
    illumination_profiles_folder: Path,
    n_images: int = 128,
    overwrite: bool = False,
    random_seed: int = None,
    basic_smoothness: float = 1,
    client=None,
) -> dict:

    """
    Calculates illumination correction profiles based on a random sample
    of images for each channel.
    NOTE: This assumes, that all wells contain the same number of channels and
    components.

    Args:
        input_zarr: input path where the image data is stored as
            OME-Zarr. Should point to the *.zarr folder.
        illumination_profiles_folder: Path to folder where illumination
            profiles will be saved.
        n_images: Number of images to sample for illumination correction.
        overwrite: If True, overwrite existing illumination profiles.
        random_seed: integer random seed to initialize random number generator.
            None will result in non-reproducibel outputs
    """
    
    logger.info(f"Calculating illumination profiles based on {n_images} randomly sampled images.")

    random.seed(random_seed)

    zarrurl = input_zarr.as_posix()
    group = zarr.open_group(zarrurl, mode="r+")
    wells = [well_dict['path'] for well_dict in group.attrs['plate']["wells"]]

    # get list of all channels
    well_path = group.attrs['plate']["wells"][0]["path"]
    well_group = zarr.open_group(f"{zarrurl}/{well_path}", mode="r+")

    channel_acquisition_dict = {}
    for acquisition in well_group.attrs['well']["images"]:
        omero_channels = get_omero_channel_list(
            image_zarr_path=f"{zarrurl}/{well_path}/{acquisition['path']}")
        channel_acquisition_dict.update(
            {channel.label: acquisition["path"] for channel in omero_channels})

    basic_dict = {}
    for channel, acquisition in channel_acquisition_dict.items():
        logger.info(
            f"Calculating illumination profile for channel {channel} & acquisition {acquisition}.")
        
        # get a list of all FOVs from all wells (image_loc_all)
        image_loc_all = []
        for well in wells:
            image_zarr_path = f"{zarrurl}/{well}/{acquisition}"
            # Read attributes from NGFF metadata
            ngff_image_meta = load_NgffImageMeta(image_zarr_path)
            coarsening_xy = ngff_image_meta.coarsening_xy
            full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
            FOV_ROI_table = ad.read_zarr(
                    f"{image_zarr_path}/tables/FOV_ROI_table")
            list_indices = convert_ROI_table_to_indices(
                FOV_ROI_table,
                level=0,
                coarsening_xy=coarsening_xy,
                full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
            )
            check_valid_ROI_indices(list_indices, "FOV_ROI_table")
            for indices in list_indices:
                image_loc_all.append((well, indices))

        # choose subset of all wells & FOVs
        image_loc_sample = random.sample(image_loc_all, n_images)

        # cycle through all samples and load data
        ROI_data = []
        for well, indices in image_loc_sample:
            image_zarr_path = f"{zarrurl}/{well}/{acquisition}"
            tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=image_zarr_path,
                wavelength_id=None,
                label=channel,
            )
            ind_channel = tmp_channel.index
            data_zyx = da.from_zarr(f"{image_zarr_path}/{acquisition}")[ind_channel]
            # Define region
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            ROI_data.append(data_zyx[region])
        
        ROI_data = da.stack(ROI_data, axis=0).compute()

        # calculate illumination correction profile
        logger.info(f"Now calculating illumination correction for channel {channel}.")
        basic = BaSiC(
            get_darkfield=True,
            smoothness_flatfield=basic_smoothness,
            smoothness_darkfield=basic_smoothness)
        if np.shape(ROI_data)[0] == 1:
            basic.fit(ROI_data[0, :, :, :])
        else:
            basic.fit(np.squeeze(ROI_data))
        logger.info(
            f"Finished calculating illumination correction for channel {channel}.")

        # save illumination correction model
        logger.info(f"Now saving illumination correction model for channel {channel}.")
        folder_path = Path(illumination_profiles_folder) / f"{channel}"
        if overwrite:
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=False)
        #basic.save_model(model_dir=filename, overwrite=overwrite)
        np.save(folder_path / 'flatfield.npy', basic.flatfield)
        np.save(folder_path / 'darkfield.npy', basic.darkfield)
        np.save(folder_path / 'baseline.npy', basic.baseline)
        basic_dict[channel] = basic
    
    return basic_dict

# if __name__ == "__main__":
#     from fractal_tasks_core.tasks._utils import run_fractal_task

#     run_fractal_task(
#         task_function=calculate_illumination_profiles,
#         logger_name=logger.name,
#     )