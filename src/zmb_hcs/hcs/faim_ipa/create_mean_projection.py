import logging
from pathlib import Path
from typing import Union

import dask
import numpy as np
import tifffile
from PIL import Image, TiffImagePlugin
from zmb_hcs.parser_MD import parse_files_zmb

logger = logging.getLogger(__name__)

def _save_mean_projection(files):
    # TODO: can maybe increase efficiency by not passing dataframe, but paths
    # calculate mean projection
    images = []
    for fn in files.query("z.notnull()").path:
        with tifffile.TiffFile(fn) as tif:
            images.append(tif.pages[0].asarray())
    images = np.stack(images)
    projection = np.mean(images, axis=0).astype(dtype=images.dtype)

    # get metadata from already existing MIP tiff
    # TODO: handle case if MIP is not present
    assert len(files.query("z.isnull()")) == 1
    file = files.query("z.isnull()").iloc[0]
    old_path = file.path
    new_path = file.new_path
    im = Image.open(old_path)
    tiffinfo = im.tag_v2
    del tiffinfo[
        TiffImagePlugin.ROWSPERSTRIP
    ]  # somehow PIL has problems with this tag -> remove it

    # change image metadata
    imageDescription = tiffinfo[TiffImagePlugin.IMAGEDESCRIPTION]
    imageDescription = imageDescription.replace(
        "Maximum", "Mean"
    )  # TODO: do this in a better way
    tiffinfo[TiffImagePlugin.IMAGEDESCRIPTION] = imageDescription

    # save projection with new metadata
    new_im = Image.fromarray(projection)
    Path(new_path).parent.mkdir(parents=True, exist_ok=True)
    new_im.save(new_path, tiffinfo=tiffinfo)

    # close images
    im.close()
    new_im.close()


def create_mean_projection(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    query: str = None,
) -> None:
    """
    Takes an MD-ImageXpress data-folder that contains multiple z-planes
    and calculates the mean projections of it.

    An new data-folder with the same structure will be created in output_path,
    with the projections stored in ZStep_0.

    If the output_path = input_path, the projections will be stored in ZStep_0.
    If the data-folder already contains a projection, it will be overwritten.

    TODO: currently the input folder needs to already contain a maximum intensity
    projection. -> should be generalized

    Args:
        input_path: Input folder of MD-ImageXpress data
        output_path: Path to the output folder
        query: Pandas-query to filter input files

    Returns:
        None
    """

    logger.info("Start 'create_mean_projection'")

    logger.info("Parsing files")
    files, mode = parse_files_zmb(input_path)
    if query is not None:
        files = files.query(query).copy()

    # define output paths
    # TODO: handle case if MIP is not present
    for index, row in files.query("z.isnull()").iterrows():
        new_path = Path(output_path) / Path(row.path).relative_to(input_path)
        files.loc[index, "new_path"] = str(new_path)

    delayed_list = []
    for well in files.well.unique():
        well_files = files.query("well==@well")
        for field in well_files.field.unique():
            field_files = well_files.query("field==@field")
            for channel in field_files.channel.unique():
                channel_files = field_files.query("channel==@channel")
                delayed_list.append(dask.delayed(_save_mean_projection)(channel_files))

    logger.info(f"Calculating and saving {len(delayed_list)} projections...")
    chunk_len = 20000
    delayed_list_chunks = [
        delayed_list[x : x + chunk_len] for x in range(0, len(delayed_list), chunk_len)
    ]
    for i, delayed_list_chunk in enumerate(delayed_list_chunks):
        logger.info(
            f"processing tiles {i*chunk_len}-{min((i+1)*chunk_len, len(delayed_list))} of {len(delayed_list)}"
        )
        _ = dask.compute(*delayed_list_chunk)
    logger.info("Finished!")
