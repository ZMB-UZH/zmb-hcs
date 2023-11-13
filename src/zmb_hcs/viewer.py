import numpy as np
import dask.array as da
#import glob
#import matplotlib.pyplot as plt
import tifffile
#import m2stitch
#import time
#from scipy.ndimage import distance_transform_edt

from .parser_MD import parse_files_zmb, get_well_image_FCZYX
from .stitching import fuse_mean

def load_for_viewing_MD(
    dir_import,
    tile_subset=None,
    MIP=False,
    fused=True,
    fuse_fun=fuse_mean,
):
    # TODO: right now, already saved projections are ignored...

    # parse files
    files, mode = parse_files_zmb(dir_import)
    files = files[files.z.notnull()]
    
    # get channel_names
    channels = files.channel.unique()
    channel_names = []
    for channel in channels:
        files_channel = files.query("channel==@channel")
        fn_sel = list(files_channel['path'])[0]
        with tifffile.TiffFile(fn_sel) as tif:
            metadata = tif.metaseries_metadata
            channel_name = metadata['PlaneInfo']['_IllumSetting_']
            channel_names.append(channel_name)
        
    # load data as dask-array
    data_da, stage_positions_da, dx, dy = get_well_image_FCZYX(files, channels)
    if tile_subset != None:
        data_da = data_da[tile_subset]
        stage_positions_da = stage_positions_da[tile_subset]
    
    if fused==False:
        if MIP==False:
            return data_da
        else:
            return data_da.max(axis=2)

    # load stage positions
    stage_positions_subset = stage_positions_da[:,0,0].compute()
    positions = stage_positions_subset[:,(1,0)]
    positions = np.round(positions / (dx, dy)).astype(int)
    positions -= positions.min(axis=0)
    nx_tot, ny_tot = positions.max(axis=0) + data_da.shape[-2:]
    
    # fusing   
    if MIP==False:
        def _fuse_xy(x, block_id=None):
            tiles = x[:,0,0]
            im_fused = fuse_fun(tiles, positions)
            return np.array([[im_fused]])
    
        imgs_fused_da = da.map_blocks(
            _fuse_xy,
            data_da,
            chunks = da.core.normalize_chunks(
                (1,1,nx_tot,ny_tot),
                shape = data_da.shape[1:-2]+(nx_tot,ny_tot)
            ),
            drop_axis = 0,
            meta = np.asanyarray([]).astype(data_da.dtype)
        )
    else:
        def _fuse_xy(x, block_id=None):
            tiles = x[:,0]
            im_fused = fuse_fun(tiles, positions)
            return np.array([im_fused])
    
        imgs_fused_da = da.map_blocks(
            _fuse_xy,
            data_da.max(axis=2),
            chunks = da.core.normalize_chunks(
                (1,nx_tot,ny_tot),
                shape = data_da.max(axis=2).shape[1:-2]+(nx_tot,ny_tot)
            ),
            drop_axis = 0,
            meta = np.asanyarray([]).astype(data_da.dtype)
        )
    
    return imgs_fused_da