import numpy as np
import pandas as pd
import dask.array as da
from .parser_MD import get_well_image_FCZYX

def get_random_slice(channel_name, files):
    data_slc_da_list = []
    for plate_dir in files.plate_dir.unique():
        plate_files = files.query("plate_dir==@plate_dir")
        for well in plate_files.well.unique():
            well_files = plate_files.query("well==@well")
            channel_files = well_files.query("channel_name==@channel_name")
            channels = channel_files.channel.unique()
            if len(channels) > 0:
                data_da, stage_positions_da, dx, dy = get_well_image_FCZYX(channel_files, channels)
                rand_slc = np.random.randint(data_da.shape[2])
                data_slc_da_list.append(data_da[:,0,rand_slc])
    data_slc_da = da.concatenate(data_slc_da_list, axis=0)
    return data_slc_da, dx, dy

def get_middle_slice(channel_name, files):
    data_slc_da_list = []
    for plate_dir in files.plate_dir.unique():
        plate_files = files.query("plate_dir==@plate_dir")
        for well in plate_files.well.unique():
            well_files = plate_files.query("well==@well")
            channel_files = well_files.query("channel_name==@channel_name")
            channels = channel_files.channel.unique()
            if len(channels) > 0:
                data_da, stage_positions_da, dx, dy = get_well_image_FCZYX(channel_files, channels)
                mid_slc = data_da.shape[2]//2
                data_slc_da_list.append(data_da[:,0,mid_slc])
    data_slc_da = da.concatenate(data_slc_da_list, axis=0)
    return data_slc_da, dx, dy

def get_mean_of_slices(channel_name, files):
    data_mnp_da_list = []
    for plate_dir in files.plate_dir.unique():
        plate_files = files.query("plate_dir==@plate_dir")
        for well in plate_files.well.unique():
            well_files = plate_files.query("well==@well")
            channel_files = well_files.query("channel_name==@channel_name")
            channels = channel_files.channel.unique()
            if len(channels) > 0:
                data_da, stage_positions_da, dx, dy = get_well_image_FCZYX(channel_files, channels)
                data_mnp_da_list.append(data_da.mean(axis=(1,2)))
    data_mnp_da = da.concatenate(data_mnp_da_list, axis=0)
    return data_mnp_da, dx, dy