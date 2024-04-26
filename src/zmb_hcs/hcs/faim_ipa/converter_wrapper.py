from typing import Union
from pathlib import Path

from dask.distributed import Client

from zmb_hcs.hcs.faim_ipa.imagexpressZMB import StackAcquisition
from zmb_hcs.hcs import roi_tables
from faim_ipa.hcs.acquisition import TileAlignmentOptions
from faim_ipa.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_ipa.hcs.plate import PlateLayout
from faim_ipa.stitching import stitching_utils
from fractal_tasks_core.tables import write_table


def convert_MD(
        *,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        zarr_name: str,
        order_name: str = 'some-order',
        barcode: str = 'some-barcode',
        well_sub_group: str = '0',
        layout: PlateLayout = PlateLayout.I96,
        query: str = None,
        alignment: TileAlignmentOptions = TileAlignmentOptions.GRID,
        client: Client = None,
        chunks: Union[tuple[int, int], tuple[int, int, int]] = (1, 2048, 2048),
) -> dict:
    """
    Create OME-Zarr plate from MD Image Xpress files.

    This is a non-parallel task => it parses the metadata, creates the plates
    and then converts all the wells in the same process

    Args:
        input_path: Input folder of MD-ImageXpress data
        output_path: Path to the output file
        zarr_name: Name of the zarr plate file that will be created
        order_name: Name of the order
        barcode: Barcode of the plate
        well_sub_group: name of image in well
        layout: Plate layout for the Zarr file. Valid options are
            PlateLayout.I18
            PlateLayout.I24
            PlateLayout.I96
            PlateLayout.I384
        query: Pandas-query to filter input files
        alignment: TileAlignmentOptions. Valid options are
            TileAlignmentOptions.STAGE_POSITION
            TileAlignmentOptions.GRID
        client: Dask client used for the conversion.
        chunks: Chunk size in (Z)YX.

    Returns:
        Metadata dictionary
    """

    print("Parsing files...")
    plate_acquisition = StackAcquisition(
        acquisition_dir=input_path,
        alignment=alignment,
        query=query,
    )

    converter = ConvertToNGFFPlate(
        ngff_plate=NGFFPlate(
            root_dir=output_path,
            name=zarr_name,
            layout=layout,
            order_name=order_name,
            barcode=barcode,
        ),
        yx_binning=1,
        stitching_yx_chunk_size_factor=1,
        warp_func=stitching_utils.translate_tiles_2d,
        fuse_func=stitching_utils.fuse_mean,
        client=client,
    )

    plate = converter.create_zarr_plate(plate_acquisition)

    # Run conversion.
    print("Converting data...")
    converter.run(
        plate=plate,
        plate_acquisition=plate_acquisition,
        well_sub_group=well_sub_group,
        chunks=chunks,
        max_layer=5,
    )

    # write ROI tables
    plate_name = zarr_name + ".zarr"
    well_paths = []
    image_paths = []

    well_acquisitions = plate_acquisition.get_well_acquisitions(selection=None)
    ROI_tables = roi_tables.create_ROI_tables(plate_acquistion=plate_acquisition)

    for well_acquisition in well_acquisitions:
        well_rc = well_acquisition.get_row_col()
        well_path = f"{plate_name}/{well_rc[0]}/{well_rc[1]}"
        well_paths.append(well_path)

        image_path = f"{well_path}/{well_sub_group}"
        image_paths.append(image_path)

        # Write the tables
        image_group = plate[well_rc[0]][well_rc[1]][well_sub_group]
        tables = ROI_tables[well_acquisition.name].keys()
        for table_name in tables:
            write_table(
                image_group=image_group,
                table_name=table_name,
                table=ROI_tables[well_acquisition.name][table_name],
                overwrite=True,
                table_type="roi_table",
                table_attrs=None,
            )

        metadata = {
            "plate": [plate_name],
            "well": well_paths,
            "image": image_paths,
        }

    return metadata