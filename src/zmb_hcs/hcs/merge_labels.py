import shutil
from pathlib import Path
from typing import Optional, Sequence

import zarr


# TODO: Add overwrite option
def merge_labels(
    zarr_url_origin: str,
    zarr_url_target: str,
    label_names_to_copy: Optional[Sequence[str]] = None,
) -> None:
    """
    Copy labels from one OME-Zarr image to another.

    Args:
        zarr_url_origin: Path or url to the individual OME-Zarr image, where
            the labels are located.
        zarr_url_target: Path or url to the individual OME-Zarr image, where
            the labels should be copied to.
        label_names_to_copy: List of label names to copy. If None, all labels
            are copied.
    """
    plate_name_origin = Path(zarr_url_origin.split(".zarr/")[0]).name
    component_origin = zarr_url_origin.split(".zarr/")[1]
    plate_name_target = Path(zarr_url_target.split(".zarr/")[0]).name
    component_target = zarr_url_target.split(".zarr/")[1]

    # check that components are the same
    if component_origin != component_target:
        raise ValueError("The origin and target components must be the same.")

    # load image groups
    image_group_origin = zarr.group(zarr_url_origin)
    image_group_target = zarr.group(zarr_url_target)

    # load origin labels
    if "labels" not in set(image_group_origin.group_keys()):
        raise ValueError("The origin does not have a labels group.")
    labels_group_origin = image_group_origin["labels"]
    label_names_origin = labels_group_origin.attrs.asdict().get("labels", [])

    # load target labels
    if "labels" not in set(image_group_target.group_keys()):
        labels_group_target = image_group_target.create_group("labels", overwrite=False)
    else:
        labels_group_target = image_group_target["labels"]
    label_names_target = labels_group_target.attrs.asdict().get("labels", [])

    # check which labels to copy
    if label_names_to_copy is None:
        label_names_to_copy = label_names_origin

    # copy labels
    for label_name in label_names_to_copy:
        if label_name not in label_names_origin:
            raise ValueError(f"Label {label_name} not found in {zarr_url_origin}.")
        if label_name in label_names_target:
            raise ValueError(f"Label {label_name} already exists in {zarr_url_target}.")
        # copy label:
        new_labels = [*label_names_target, label_name]
        labels_group_target.attrs["labels"] = new_labels

        path_origin = Path(zarr_url_origin) / "labels" / label_name
        path_target = Path(zarr_url_target) / "labels" / label_name
        # os.system(f"cp -r {str(path_origin)}  {str(path_target)}")
        shutil.copytree(str(path_origin), str(path_target))
