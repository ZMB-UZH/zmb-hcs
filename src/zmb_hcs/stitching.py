import numpy as np
import dask.array as da
import glob
import matplotlib.pyplot as plt
import tifffile
import m2stitch
import time
from scipy.ndimage import distance_transform_edt
from aicsimageio import AICSImage

from .parser_MD import parse_files_zmb, get_well_image_FCZYX


def get_grid_params(rows_or_cols, threshold):
    sorted = np.sort(rows_or_cols)
    n = 0
    bins = [[]]
    prev = 0.
    for el in sorted:
        if el-prev < threshold:
            bins[n].append(el)
            prev = el
        else:
            n += 1
            bins.append([])
            bins[n].append(el)
            prev = el
    means = [np.mean(bin) for bin in bins]
    start = means[0]
    delta = np.mean(np.diff(means))
    return start, delta


def construct_grid(positions, tile_dimensions):
    start_x, delta_x = get_grid_params(positions[:,0], tile_dimensions[0]*0.5)
    start_y, delta_y = get_grid_params(positions[:,1], tile_dimensions[1]*0.5)
    index_x = np.round((positions[:,0] - start_x) / delta_x).astype(int) + 1
    index_y = np.round((positions[:,1] - start_y) / delta_y).astype(int) + 1
    return (index_x, index_y)


def check_tile_configuration(dir_import):
    files, mode = parse_files_zmb(dir_import)
    print(f'Total number of files: {len(files)}')

    # set projections from 'None' to 0
    files.loc[files.z.isna(),'z'] = 0
    
    data_all_da, stage_positions_all_da, dx, dy = get_well_image_FCZYX(
        files,
        files.channel.unique()
    )
    
    stage_positions_flat = stage_positions_all_da[:,0,0].compute()
    positions = stage_positions_flat[:,(1,0)]
    positions = np.round(positions / (dx, dy)).astype(int)
    positions -= positions.min(axis=0)
    
    fig, ax = plt.subplots()
    xs = positions[:,1]
    ys = positions[:,0]
    ax.plot(xs, ys, "-o")
    ax.invert_yaxis()
    for n, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, str(n), color="red", fontsize=12)
    plt.show()


def fuse_mean(tiles, positions):
    nx_tot, ny_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((nx_tot, ny_tot), dtype=tiles.dtype)
    im_count = np.zeros_like(im_fused, dtype='uint8')
    tile_count = np.ones_like(tiles[0], dtype='uint8')
    nx_tile, ny_tile = tiles.shape[-2:]
    
    for tile, pos in zip(tiles, positions):
        im_fused[pos[0]:pos[0]+nx_tile, pos[1]:pos[1]+ny_tile] += tile
        im_count[pos[0]:pos[0]+nx_tile, pos[1]:pos[1]+ny_tile] += tile_count
    
    with np.errstate(divide='ignore'):
        im_fused = im_fused//im_count
    return im_fused

def fuse_mean_gradient(tiles, positions):
    nx_tot, ny_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((nx_tot, ny_tot), dtype='uint32')
    im_weight = np.zeros_like(im_fused, dtype='uint32')
    nx_tile, ny_tile = tiles.shape[-2:]
    
    # distance map to border of image
    mask = np.ones((nx_tile, ny_tile))
    mask[[0,-1],:] = 0
    mask[:,[0,-1]] = 0
    dist_map = distance_transform_edt(mask).astype('uint32') + 1
    
    for tile, pos in zip(tiles, positions):
        im_fused[pos[0]:pos[0]+nx_tile, pos[1]:pos[1]+ny_tile] += tile*dist_map
        im_weight[pos[0]:pos[0]+nx_tile, pos[1]:pos[1]+ny_tile] += dist_map
    with np.errstate(divide='ignore'):
        im_fused = im_fused//im_weight
    return im_fused.astype(tiles.dtype)

def fuse_random_pixel_gradient(tiles, positions):
    nx_tot, ny_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((len(tiles), nx_tot, ny_tot), dtype=tiles.dtype)
    im_weight = np.zeros_like(im_fused, dtype='uint16')
    nx_tile, ny_tile = tiles.shape[-2:]
    
    # distance map to border of image
    mask = np.ones((nx_tile, ny_tile))
    mask[[0,-1],:] = 0
    mask[:,[0,-1]] = 0
    dist_map = distance_transform_edt(mask).astype('uint32') + 1
    
    for i in range(len(tiles)):
        tile = tiles[i]
        pos = positions[i]
        im_fused[i, pos[0]:pos[0]+nx_tile, pos[1]:pos[1]+ny_tile] += tile
        im_weight[i, pos[0]:pos[0]+nx_tile, pos[1]:pos[1]+ny_tile] += dist_map
    im_weight[:,im_weight.max(axis=0) == 0] = 1
    im_weight = im_weight / im_weight.sum(axis=0)[np.newaxis,:,:]
    im_weight = np.cumsum(im_weight, axis=0)
    im_weight = np.insert(im_weight, 0, np.zeros_like(im_weight[0]), axis=0)
    im_rand = np.random.rand(*im_fused[0].shape)
    for i in range(len(im_fused)):
        im_fused[i,(im_rand < im_weight[i]) | (im_weight[i+1] < im_rand)] = 0
    return im_fused.sum(axis=0)

def fuse_fw(tiles, positions):
    nx_tot, ny_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((nx_tot, ny_tot), dtype=tiles.dtype)
    nx_tile, ny_tile = tiles.shape[-2:]
    
    for tile, pos in zip(tiles, positions):
        im_fused[pos[0]:pos[0]+nx_tile, pos[1]:pos[1]+ny_tile] = tile
        
    return im_fused

def fuse_rev(tiles, positions):
    nx_tot, ny_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((nx_tot, ny_tot), dtype=tiles.dtype)
    nx_tile, ny_tile = tiles.shape[-2:]
    
    for tile, pos in reversed(list(zip(tiles, positions))):
        im_fused[pos[0]:pos[0]+nx_tile, pos[1]:pos[1]+ny_tile] = tile
        
    return im_fused


def stitch_and_export_MD(
    save_name,
    dir_import,
    tile_subset=None,
    channels_used_for_stitching=[0,],
    flatfield_directory=None,
    fuse_fun=fuse_mean_gradient,
):
    print('----------------------------------------------')
    print('Processing:')
    print(f'Save-Name: {save_name}')
    print(f'Directory: {dir_import}')
    if tile_subset != None:
        print(f'Tiles: {tile_subset}')
        
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
        
    print('\nLoading stage positions...')
    start_time = time.time()
    stage_positions = stage_positions_da.compute()
    dz = np.round(
        np.mean(stage_positions[:,0,1:,2]-stage_positions[:,0,:-1,2]),
        2
    )
    positions = stage_positions[:,0,0,(1,0)]
    positions = np.round(positions / (dx, dy)).astype(int)
    positions -= positions.min(axis=0)
    print(f"took {(time.time() - start_time):.1f}s")
    print(f'dz = {dz}')
    
    # load flatfield images
    if flatfield_directory != None:
        flatfields = np.empty((data_da.shape[1],)+data_da.shape[-2:])
        for c, channel_name in enumerate(channel_names):
            fn_ff = glob.glob(flatfield_directory+f"\\*{channel_name}*.tif")[0]
            with tifffile.TiffFile(fn_ff) as tif:
                image = tif.pages[0].asarray()
                flatfields[c] = image

    # stitching
    print('\nCalculating MIP for stitching...')
    start_time = time.time()
    mip_da = da.max(data_da, axis=2)
    if flatfield_directory != None:
        mip_da = mip_da / flatfields
    mip_da = da.mean(mip_da[:,channels_used_for_stitching], axis=1)
    mip = mip_da.compute()
    print(f"took {(time.time() - start_time):.1f}s")

    print('\nStitching...')
    start_time = time.time()
    rows, cols = construct_grid(positions, data_da.shape[-2:])
    result_df, _ = m2stitch.stitch_images(
        mip, rows, cols, row_col_transpose=False, ncc_threshold=0.1
    )
    print(f"took {(time.time() - start_time):.1f}s")

    # fusing     
    def _fuse_xy(x, block_id=None):
        tiles = x[:,0,0]
        if flatfield_directory != None:
            channel = block_id[0]
            tiles = (tiles/flatfields[channel]).astype(x.dtype)
        im_fused = fuse_fun(tiles, positions_stitched)
        return np.array([[im_fused]])

    print('\nFusing images...')
    start_time = time.time()
    positions_stitched = result_df.to_numpy()[:,2:]
    positions_stitched = positions_stitched - positions_stitched.min(axis=0)
    nx_tot, ny_tot = positions_stitched.max(axis=0) + data_da.shape[-2:]
    
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

    imgs_fused = imgs_fused_da.compute()
    print(f"took {(time.time() - start_time):.1f}s")

    print('\nSaving images...')
    start_time = time.time()
    for c, stack in enumerate(imgs_fused):
        fn = save_name + f'_{channel_names[c]}.ome.tif'
        with tifffile.TiffWriter(fn, bigtiff=True) as tif:
            tif.write(stack, photometric='minisblack', metadata={
                'axes': 'ZYX',
                'PhysicalSizeX': dx,
                'PhysicalSizeY': dy,
                'PhysicalSizeZ': dz,
                'Plane': {'PositionX': [stage_positions[0, 0, 0, 0]
                          for n in range(stack.shape[0])],
                          'PositionY': [stage_positions[0, 0, 0, 1]
                          for n in range(stack.shape[0])],
                          'PositionZ': [stage_positions[0, 0, 0, 2] + n
                          * dz for n in range(stack.shape[0])]},
                })
    print(f"took {(time.time() - start_time):.1f}s\n")


def stitch_and_export_CZI(
    save_name,
    import_fn,
    tile_subset=None,
    channels_used_for_stitching=[0,],
    flatfield_directory=None,
    fuse_fun=fuse_mean_gradient,
):
    print('----------------------------------------------')
    print('Processing:')
    print(f'Save-Name: {save_name}')
    print(f'File: {import_fn}')
    if tile_subset != None:
        print(f'Tiles: {tile_subset}')

    reader = AICSImage(import_fn, reconstruct_mosaic=False)

    # load data as dask-array
    # note: get_image_dask_data() should return 'zyx' as one chunk
    data_da = reader.get_image_dask_data("MCZYX", T=0)
    if tile_subset != None:
        data_da = data_da[tile_subset]
        
    # load some metadata
    channel_names = reader.channel_names
    dx = reader.physical_pixel_sizes.X
    dy = reader.physical_pixel_sizes.Y
    dz = reader.physical_pixel_sizes.Z

    # load stage positions
    print('\nLoading stage positions...')
    start_time = time.time()
    stage_positions = np.array(reader.get_mosaic_tile_positions(T=0,C=0,Z=0))
    if tile_subset != None:
        stage_positions = stage_positions[tile_subset]
    positions = np.round(stage_positions / (dx, dy)).astype(int)
    positions -= positions.min(axis=0)
    print(f"took {(time.time() - start_time):.1f}s")
        
    # load flatfield images
    if flatfield_directory != None:
        flatfields = np.empty((data_da.shape[1],)+data_da.shape[-2:])
        for c, channel_name in enumerate(channel_names):
            fn_ff = glob.glob(flatfield_directory+f"\\*{channel_name}*.tif")[0]
            with tifffile.TiffFile(fn_ff) as tif:
                image = tif.pages[0].asarray()
                flatfields[c] = image

    # stitching
    print('\nCalculating MIP for stitching...')
    start_time = time.time()
    mip_da = da.max(data_da, axis=2)
    if flatfield_directory != None:
        mip_da = mip_da / flatfields
    mip_da = da.mean(mip_da[:,channels_used_for_stitching], axis=1)
    mip = mip_da.compute()
    print(f"took {(time.time() - start_time):.1f}s")

    print('\nStitching...')
    start_time = time.time()
    rows, cols = construct_grid(positions, data_da.shape[-2:])
    result_df, _ = m2stitch.stitch_images(
        mip, rows, cols, row_col_transpose=False, ncc_threshold=0.1
    )
    print(f"took {(time.time() - start_time):.1f}s")

    # fusing     
    def _fuse_xy(x, block_id=None):
        tiles = x[:,0,0]
        if flatfield_directory != None:
            channel = block_id[0]
            tiles = (tiles/flatfields[channel]).astype(x.dtype)
        im_fused = fuse_fun(tiles, positions_stitched)
        return np.array([[im_fused]])

    print('\nFusing images...')
    start_time = time.time()
    positions_stitched = result_df.to_numpy()[:,2:]
    positions_stitched = positions_stitched - positions_stitched.min(axis=0)
    nx_tot, ny_tot = positions_stitched.max(axis=0) + data_da.shape[-2:]
    
    data_da = da.rechunk(data_da, chunks=((1,1,1,)+data_da.shape[-2:]))
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

    imgs_fused = imgs_fused_da.compute()
    print(f"took {(time.time() - start_time):.1f}s")

    print('\nSaving images...')
    start_time = time.time()
    for c, stack in enumerate(imgs_fused):
        fn = save_name + f'_{channel_names[c]}.ome.tif'
        with tifffile.TiffWriter(fn, bigtiff=True) as tif:
            tif.write(stack, photometric='minisblack', metadata={
                'axes': 'ZYX',
                'PhysicalSizeX': dx,
                'PhysicalSizeY': dy,
                'PhysicalSizeZ': dz,
                'Plane': {'PositionX': [stage_positions[0, 0]
                          for n in range(stack.shape[0])],
                          'PositionY': [stage_positions[0, 1]
                          for n in range(stack.shape[0])],
                          #TODO: CZI: see, if we can read in actula z-position
                          'PositionZ': [0 + n
                          * dz for n in range(stack.shape[0])]},
                })
    print(f"took {(time.time() - start_time):.1f}s\n")
