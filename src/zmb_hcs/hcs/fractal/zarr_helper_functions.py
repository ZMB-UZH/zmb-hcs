import zarr
from pathlib import Path

# deprecated for fractal 2.0
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

def get_image_urls(
    zarr_path: Path,
) -> list[str]:
    zarrurl = zarr_path.as_posix()
    group = zarr.open_group(zarrurl, mode="r+")
    images = []
    for well in group.attrs['plate']['wells']:
        well_path = f"{zarrurl}/{well['path']}"
        well_group = zarr.open_group(well_path, mode="r+")
        for image in well_group.attrs['well']['images']:
            images.append(f"{well_path}/{image['path']}")
    return images