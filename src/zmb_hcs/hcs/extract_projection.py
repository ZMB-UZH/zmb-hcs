import logging
from pathlib import Path
from typing import Union

import dask
import numpy as np
import tifffile
from PIL import Image, TiffImagePlugin
from zmb_hcs.parser_MD import parse_files_zmb

logger = logging.getLogger(__name__)

def _calculate_and_save_projection(files, projection_type):
    # TODO: can maybe increase efficiency by not passing dataframe, but paths
    # calculate projection
    images = []
    for fn in files.query("z.notnull()").path:
        with tifffile.TiffFile(fn) as tif:
            images.append(tif.pages[0].asarray())
    images = np.stack(images)
    if projection_type == "max":
        projection = np.max(images, axis=0).astype(dtype=images.dtype)
    elif projection_type == "mean":
        projection = np.mean(images, axis=0).astype(dtype=images.dtype)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}. Must be 'max' or 'mean'.")

    if len(files.query("z.isnull()")) == 1:
        # get metadata from already existing MIP tiff
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
        if projection_type == "max":
            if "<custom-prop id=\"Z Projection Method\" type=\"string\" value=\"Mean\"/>\n" in imageDescription:
                imageDescription = imageDescription.replace(
                    "<custom-prop id=\"Z Projection Method\" type=\"string\" value=\"Mean\"/>\n", "<custom-prop id=\"Z Projection Method\" type=\"string\" value=\"Maximum\"/>\n"
                )  # TODO: do this in a better way
                tiffinfo[TiffImagePlugin.IMAGEDESCRIPTION] = imageDescription
            else:
                logger.warning("Cannot find projection type in image description, but proceeding anyway")
        elif projection_type == "mean":
            if "<custom-prop id=\"Z Projection Method\" type=\"string\" value=\"Maximum\"/>\n" in imageDescription:
                imageDescription = imageDescription.replace(
                    "<custom-prop id=\"Z Projection Method\" type=\"string\" value=\"Maximum\"/>\n", "<custom-prop id=\"Z Projection Method\" type=\"string\" value=\"Mean\"/>\n"
                )  # TODO: do this in a better way
                tiffinfo[TiffImagePlugin.IMAGEDESCRIPTION] = imageDescription
            else:
                logger.warning("Cannot find projection type in image description, but proceeding anyway")
    elif len(files.query("z.isnull()")) == 0:
        # get metadata from already existing plane tiff
        file = files.query("z == '1'").iloc[0]
        old_path = file.path
        new_path = file.new_path
        im = Image.open(old_path)
        tiffinfo = im.tag_v2
        del tiffinfo[
            TiffImagePlugin.ROWSPERSTRIP
        ]  # somehow PIL has problems with this tag -> remove it

        # change image metadata
        imageDescription = tiffinfo[TiffImagePlugin.IMAGEDESCRIPTION]
        if projection_type == "max":
            imageDescription = imageDescription.replace(
                "<custom-prop id=\"Z Step\" type=\"float\" value=\"1\"/>",
                "<custom-prop id=\"Z Projection Method\" type=\"string\" value=\"Maximum\"/>\n"
                "<custom-prop id=\"Z Projection Step Size\" type=\"float\" value=\"0\"/>\n"
                "<custom-prop id=\"Z Thickness\" type=\"float\" value=\"0\"/>"
            )  # TODO: do this in a better way
        elif projection_type == "mean":
            imageDescription = imageDescription.replace(
                "<custom-prop id=\"Z Step\" type=\"float\" value=\"1\"/>",
                "<custom-prop id=\"Z Projection Method\" type=\"string\" value=\"Mean\"/>\n"
                "<custom-prop id=\"Z Projection Step Size\" type=\"float\" value=\"0\"/>\n"
                "<custom-prop id=\"Z Thickness\" type=\"float\" value=\"0\"/>"
            )
        tiffinfo[TiffImagePlugin.IMAGEDESCRIPTION] = imageDescription
    else:
        raise ValueError("Multiple MIPS found")

    # save projection with new metadata
    new_im = Image.fromarray(projection)
    Path(new_path).parent.mkdir(parents=True, exist_ok=True)
    new_im.save(new_path, tiffinfo=tiffinfo)

    # close images
    im.close()
    new_im.close()

def _copy_or_calculate_projection(files, projection_type):
    if len(files.query("z.isnull()")) == 1:
        file = files.query("z.isnull()").iloc[0]
        old_path = file.path
        im = Image.open(old_path)
        tiffinfo = im.tag_v2
        del tiffinfo[
            TiffImagePlugin.ROWSPERSTRIP
        ]  # somehow PIL has problems with this tag -> remove it
        imageDescription = tiffinfo[TiffImagePlugin.IMAGEDESCRIPTION]
        if "<custom-prop id=\"Z Projection Method\" type=\"string\" value=\"Maximum\"/>\n" in imageDescription:
            if projection_type == "max":
                # copy existing MIP
                new_path = file.new_path
                Path(new_path).parent.mkdir(parents=True, exist_ok=True)
                im.save(new_path, tiffinfo=tiffinfo)
                im.close()
                return
            elif projection_type == "mean":
                im.close()
                # calculate mean projection
                _calculate_and_save_projection(files, projection_type)
                return
        elif "<custom-prop id=\"Z Projection Method\" type=\"string\" value=\"Mean\"/>\n" in imageDescription:
            if projection_type == "mean":
                # copy existing MIP
                new_path = file.new_path
                Path(new_path).parent.mkdir(parents=True, exist_ok=True)
                im.save(new_path, tiffinfo=tiffinfo)
                im.close()
                return
            elif projection_type == "max":
                im.close()
                # calculate max projection
                _calculate_and_save_projection(files, projection_type)
                return
        else:
            logger.warning("Cannot find projection type in image description, recalculating projection")
            im.close()
            _calculate_and_save_projection(files, projection_type)
            return

def extract_projection(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    projection_type: str = "mean",
    query: str = None,
) -> None:
    """
    Takes an MD-ImageXpress data-folder that contains multiple z-planes
    and calculates a projection of it.

    A new data-folder with the same structure will be created in output_path,
    with the projections stored in ZStep_0.

    If the output_path = input_path, the projections will be stored in ZStep_0.
    If the data-folder already contains a projection, it will be overwritten.

    Args:
        input_path: Input folder of MD-ImageXpress data
        output_path: Path to the output folder
        projection_type: Type of projection to calculate ('mean' or 'max')
        query: Pandas-query to filter input files

    Returns:
        None
    """

    logger.info("Start 'extract_projection'")

    logger.info("Parsing files")
    files, mode = parse_files_zmb(input_path)
    if query is not None:
        files = files.query(query).copy()

    # define output paths
    if len(files.query("z.isnull()")) > 0:
        for index, row in files.query("z.isnull()").iterrows():
            new_path = Path(output_path) / Path(row.path).relative_to(input_path)
            files.loc[index, "new_path"] = str(new_path)
    else:
        for index, row in files.query("z == '1'").iterrows():
            relpath = Path(row.path).relative_to(input_path)
            parts = list(relpath.parts)
            parts[-2] = "ZStep_0"
            new_path = Path(output_path) / Path(*parts)
            files.loc[index, "new_path"] = str(new_path)

    delayed_list = []
    for well in files.well.unique():
        well_files = files.query("well==@well")
        for field in well_files.field.unique():
            field_files = well_files.query("field==@field")
            for channel in field_files.channel.unique():
                channel_files = field_files.query("channel==@channel")
                delayed_list.append(dask.delayed(_copy_or_calculate_projection)(channel_files, projection_type))

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
