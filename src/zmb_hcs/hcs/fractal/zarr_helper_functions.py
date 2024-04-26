import zarr
from pathlib import Path


def construct_metadata(
    folder_path: Path,
    plate_name: str,
    component_subgroup: str = "0",
) -> dict:
    zarrurl = (folder_path / plate_name).as_posix()
    group = zarr.open_group(zarrurl, mode="r+")
    wells = [f"{plate_name}/{well['path']}" for well in group.attrs['plate']['wells']]
    images = [f"{well}/{component_subgroup}" for well in wells]
    metadata = {
        'plate': [plate_name],
        'well': wells,
        'image': images,
    }
    return metadata