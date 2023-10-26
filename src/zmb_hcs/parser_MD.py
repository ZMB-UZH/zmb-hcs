import os
import re
import numpy as np
import pandas as pd
import tifffile
import dask.array as da

from numpy._typing import ArrayLike
from pathlib import Path
from typing import Union


# from faim-hcs
def _list_dataset_files(
    root_dir: Union[Path, str], root_re: re.Pattern, filename_re: re.Pattern
) -> list[str]:
    files = []
    for root, _, filenames in os.walk(root_dir):
        m_root = root_re.fullmatch(root)
        if m_root:
            for f in filenames:
                m_filename = filename_re.fullmatch(f)
                if m_filename:
                    row = m_root.groupdict()
                    row |= m_filename.groupdict()
                    row["path"] = str(Path(root).joinpath(f))
                    files.append(row)
    return files


# adapted from fractal-faim-hcs
def parse_files_zmb(path: Union[Path, str], query: str=""):
    """Parse files from a Molecular Devices ImageXpress dataset for ZMB setup."""
    _METASERIES_FILENAME_PATTERN_ZMB_2D = (
        r"(?P<name>.*)_(?P<well>[A-Z]+"
        r"\d{2})_(?P<field>s\d+)*_*"
        r"(?P<channel>w[1-9]{1})*"
        r"(?!_thumb)(?P<md_id>.*)*"
        r"(?P<ext>.tif|TIF)"
    )
    _METASERIES_ZMB_PATTERN = (
        r".*[\/\\](?P<time_point>TimePoint_[0-9]*)(?:[\/\\]" r"ZStep_(?P<z>\d+))?.*"
    )
    root_pattern = _METASERIES_ZMB_PATTERN
    files = pd.DataFrame(
        _list_dataset_files(
            root_dir=path,
            root_re=re.compile(root_pattern),
            filename_re=re.compile(_METASERIES_FILENAME_PATTERN_ZMB_2D),
        )
    )

    if not query=="":
        files = files.query(query).copy()

    # mode detection
    if files['z'].isnull().all():
        mode = 'top-level'
    elif '0' in list(files['z']):
        mode = 'all'
        files.loc[files['z']=='0', 'z'] = None
    else:
        mode = 'z-steps'

    # Ensure that field and channel are not None
    if files["field"].isnull().all():
        files["field"] = files["field"].fillna("s1")
    if files["channel"].isnull().all():
        files["channel"] = files["channel"].fillna("w1")

    return files, mode


# from fractal-faim-hcs (dask)
def create_filename_structure(
    well_files: pd.DataFrame,
    channels: list[str],
) -> ArrayLike:
    """
    Assemble filenames in a numpy-array with ordering (field,channel,plane).
    This allows us to later easily map over the filenames to create a
    dask-array of the images.
    """
    planes = sorted(well_files["z"].unique(), key=int)
    fields = sorted(
        well_files["field"].unique(),
        key=lambda s: int(re.findall(r"(\d+)", s)[0])
    )

    # Create an empty np array to store the filenames in the correct structure
    fn_dtype = f"<U{max([len(fn) for fn in well_files['path']])}"
    fns_np = np.zeros(
        (len(fields), len(channels), len(planes)),
        dtype=fn_dtype,
    )

    # Store fns in correct position
    for s, field in enumerate(fields):
        field_files = well_files[well_files["field"] == field]
        for c, channel in enumerate(channels):
            channel_files = field_files[field_files["channel"] == channel]
            for z, plane in enumerate(planes):
                plane_files = channel_files[channel_files["z"] == plane]
                if len(plane_files) == 1:
                    fns_np[s, c, z] = list(plane_files["path"])[0]
                elif len(plane_files) > 1:
                    raise RuntimeError("Multiple files found for one FCZ")
    
    return fns_np


# adapted from fractal-faim-hcs (dask)
def get_well_image_FCZYX(
    well_files: pd.DataFrame,
    channels: list[str],
):
    fns_np = create_filename_structure(well_files, channels)
    
    # Get the image dimensions and some metadata from the first image
    with tifffile.TiffFile(fns_np[0,0,0]) as tif:
        image = tif.pages[0].asarray()
        metadata = tif.metaseries_metadata
    x_dim = metadata['PlaneInfo']['pixel-size-x']
    y_dim = metadata['PlaneInfo']['pixel-size-y']
    dx = metadata['PlaneInfo']['spatial-calibration-x']
    dy = metadata['PlaneInfo']['spatial-calibration-x']
    dtype = image.dtype
    
    # create dask array with fns
    fns_da = da.from_array(fns_np, chunks=(1,)*len(fns_np.shape) )
    fns_shape = fns_da.shape
    
    def _read_image(x):
        fn = x.flatten()[0]
        with tifffile.TiffFile(fn) as tif:
            image = tif.pages[0].asarray()
        newshape = x.shape + image.shape
        return np.reshape(image, newshape)

    def _read_metadata(x):
        fn = x.flatten()[0]
        with tifffile.TiffFile(fn) as tif:
            metadata = tif.metaseries_metadata
        #acqisition_time = metadata['PlaneInfo']['acquisition-time-local']
        #channel_name = metadata['PlaneInfo']['_IllumSetting_']
        x_pos = metadata['PlaneInfo']['stage-position-x']
        y_pos = metadata['PlaneInfo']['stage-position-y']
        z_pos = metadata['PlaneInfo']['z-position']
        output = np.array([x_pos, y_pos, z_pos], dtype='float32')
        newshape = x.shape + output.shape
        return np.reshape(output, newshape)

    # create dask-array for images by mapping _read_image over fns_da
    images_da = fns_da.map_blocks(
        _read_image,
        chunks=da.core.normalize_chunks( (1,)*len(fns_shape) + (x_dim,y_dim), fns_shape + (x_dim,y_dim) ),
        new_axis=list(range(len(fns_shape),len(fns_shape)+2)),
        meta=np.asanyarray([]).astype(dtype)
    )

    # create dask-array for stage-positions by mapping _read_metadata over fns_da
    stage_positions_da = fns_da.map_blocks(
        _read_metadata,
        chunks=da.core.normalize_chunks( (1,)*len(fns_shape) + (3,), fns_shape + (3,) ),
        new_axis=len(fns_shape),
        meta=np.asanyarray([]).astype('float32')
    )
    
    return images_da, stage_positions_da, dx, dy