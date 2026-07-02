"""Extract projections from MD ImageXpress HCS.ai microscope data.

This is the HCS.ai counterpart of :mod:`extract_projection`. The HCS.ai
(``MetaXpress Acquire``) format stores its metadata outside the TIFF files
(in ``image_metadata_*.csv`` and ``.jdce`` files) rather than in MetaSeries
XML headers, so parsing is delegated to the ``zmb-md-converter`` package.

The output is written as a valid HCS.ai acquisition folder: an
``experiment/`` subfolder holding the projection TIFFs, a regenerated
``image_metadata_1.csv``, a copied ``.jdce`` file, and the ``.mxprotocol``
file at the root. This lets the result be re-parsed by
``parse_MD_plate_folder`` (and hence the downstream converters).
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Optional, Union

import dask
import numpy as np
import pandas as pd
import tifffile
from zmb_md_converter.io.hcs_ai.parsing import parse_MD_plate_folder

logger = logging.getLogger(__name__)

# Normalize the various spellings the mxprotocol uses for the projection method.
_PROJECTION_MECHANISM_MAP = {
    "max": "max",
    "maximum": "max",
    "mean": "mean",
    "average": "mean",
    "avg": "mean",
}


def _find_key_values(obj, key: str) -> list:
    """Recursively collect all values stored under ``key`` in a nested dict/list."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found.extend(_find_key_values(v, key))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_find_key_values(item, key))
    return found


def _detect_microscope_projection_types(mxprotocol_path: Path) -> set:
    """Return the set of normalized projection types the microscope saved.

    Reads ``zProjectionMechanism`` entries from the ``.mxprotocol`` JSON.
    Returns an empty set if none can be determined.
    """
    try:
        with open(mxprotocol_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read projection type from {mxprotocol_path}: {e}")
        return set()

    types = set()
    for value in _find_key_values(data, "zProjectionMechanism"):
        if isinstance(value, str):
            normalized = _PROJECTION_MECHANISM_MAP.get(value.strip().lower())
            if normalized:
                types.add(normalized)
    return types


def _experiment_holds_projection(proj_csv: dict) -> bool:
    """Return True if the input ``experiment/`` folder holds computed projections.

    The microscope names projection files with a ``_projection`` infix
    (e.g. ``..._projection_t0_C05_s0_w0_z0.tif``), whereas raw single-plane or
    center-Z acquisitions keep their plain ``..._z{N}.tif`` names. The filename
    marker is therefore the reliable signal that ``experiment/`` is a real
    projection rather than raw plane data. ``zProjectionMechanism`` alone is not
    sufficient: it reads ``Max`` even for single-plane/center-Z acquisitions.
    """
    return any(
        isinstance(fname, str) and "_projection" in fname
        for fname in proj_csv
        if fname != "__columns__"
    )


def _projection_filename(stack_filename: str) -> str:
    """Turn a z-stack filename into its projection counterpart.

    ``test_data_t0_C05_s0_w0_z0.tif`` -> ``test_data_projection_t0_C05_s0_w0_z0.tif``
    matching the microscope's own naming convention.
    """
    new_name, n = re.subn(r"^(.*?)(_t\d+_)", r"\1_projection\2", stack_filename)
    if n == 0:
        # Fallback: insert before the extension if the expected pattern is absent.
        stem, dot, ext = stack_filename.rpartition(".")
        new_name = f"{stem}_projection{dot}{ext}" if dot else f"{stack_filename}_projection"
    return new_name


def _read_csv_metadata(experiment_dir: Path) -> dict:
    """Index all ``image_metadata_*.csv`` rows of an experiment dir by filename.

    Returns a dict mapping ``ImageFileName`` -> row dict, plus the list of
    columns under the special key ``"__columns__"``. Returns an empty dict if
    no CSV is present.
    """
    csv_files = sorted(experiment_dir.glob("image_metadata_*.csv"))
    if not csv_files:
        return {}
    raw = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    by_name = {row["ImageFileName"]: row.to_dict() for _, row in raw.iterrows()}
    by_name["__columns__"] = list(raw.columns)
    return by_name


def _compute_projection(paths, projection_type, slice_index):
    """Load the given plane TIFFs and reduce them along z."""
    images = np.stack([tifffile.imread(p) for p in paths])
    if projection_type == "max":
        return np.max(images, axis=0).astype(images.dtype)
    elif projection_type == "mean":
        return np.mean(images, axis=0).astype(images.dtype)
    elif projection_type == "slice":
        idx = slice_index if slice_index is not None else len(paths) // 2
        return images[idx]
    else:
        raise ValueError(
            f"Unknown projection type: {projection_type}. "
            "Must be 'max', 'mean' or 'slice'."
        )


def _copy_group(proj_row, out_experiment_dir, output_columns):
    """Copy an existing microscope projection TIFF and return its CSV row."""
    src = Path(proj_row["src_path"])
    subfolder = proj_row["ImageSubFolderPath"]
    fname = proj_row["ImageFileName"]
    dst = out_experiment_dir / subfolder / fname
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return {c: proj_row.get(c) for c in output_columns}


def _compute_group(
    plane_paths,
    template_row,
    pos_z,
    projection_type,
    slice_index,
    out_experiment_dir,
    output_columns,
):
    """Compute a projection, write the TIFF, and return its CSV row."""
    projection = _compute_projection(plane_paths, projection_type, slice_index)

    subfolder = template_row["ImageSubFolderPath"]
    out_fname = _projection_filename(template_row["ImageFileName"])
    dst = out_experiment_dir / subfolder / out_fname
    dst.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dst, projection)

    row = {c: template_row.get(c) for c in output_columns}
    row["ImageFileName"] = out_fname
    row["ZIndex"] = 0
    row["PositionZUm"] = pos_z
    if "MinIntensity" in row:
        row["MinIntensity"] = int(projection.min())
    if "MaxIntensity" in row:
        row["MaxIntensity"] = int(projection.max())
    if "MeanIntensity" in row:
        row["MeanIntensity"] = float(projection.mean())
    return row


def _copy_jdce(input_path: Path, out_experiment_dir: Path) -> None:
    """Copy the experiment .jdce into the output experiment folder.

    Prefers the input ``experiment/`` jdce (copied verbatim); otherwise derives
    it from the ``experiment_z_stack/`` jdce by stripping the ``_z_stack``
    suffix from its on-disk filename (matching the HCS.ai naming convention).

    The output name is derived from the source *filename* rather than the
    parsed plate name on purpose: the plate name can contain characters that
    are illegal in filenames (e.g. a ``:`` in a timestamp) when the JDCE's
    ``PlateId`` and ``Creation`` fields disagree, whereas an on-disk filename
    is always valid.
    """
    out_experiment_dir.mkdir(parents=True, exist_ok=True)
    proj_dir = input_path / "experiment"
    if proj_dir.is_dir():
        jdce_files = sorted(proj_dir.glob("*.jdce"))
        if jdce_files:
            shutil.copy2(jdce_files[0], out_experiment_dir / jdce_files[0].name)
            return
    z_stack_dir = input_path / "experiment_z_stack"
    if z_stack_dir.is_dir():
        jdce_files = sorted(z_stack_dir.glob("*.jdce"))
        if jdce_files:
            src = jdce_files[0]
            stem = src.stem
            if stem.endswith("_z_stack"):
                stem = stem[: -len("_z_stack")]
            shutil.copy2(src, out_experiment_dir / f"{stem}.jdce")
            return
    logger.warning(f"No .jdce file found in {input_path}; output may not re-parse.")


def extract_projection_hcs_ai(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    projection_type: str = "mean",
    slice_index: Optional[int] = None,
    query: Optional[str] = None,
) -> None:
    """Extract a projection from an MD ImageXpress HCS.ai acquisition folder.

    A new HCS.ai acquisition folder is written to ``output_path`` containing an
    ``experiment/`` subfolder with the projections, a regenerated
    ``image_metadata_1.csv``, a copied ``.jdce`` file, and the ``.mxprotocol``
    file. If the microscope already saved a projection of the requested type in
    the input ``experiment/`` folder, it is copied instead of recomputed;
    otherwise the projection is computed from ``experiment_z_stack/``.

    Args:
        input_path: HCS.ai acquisition root (the folder containing the
            ``.mxprotocol`` file).
        output_path: Path to the output acquisition root.
        projection_type: 'mean', 'max' or 'slice'.
        slice_index: For ``projection_type='slice'``, the z-index to extract.
            If None, the middle slice is used.
        query: Optional pandas query string to filter the parsed files.

    Returns:
        None
    """
    logger.info("Start 'extract_projection_hcs_ai'")
    input_path = Path(input_path)
    output_path = Path(output_path)

    mxprotocol_files = sorted(input_path.glob("*.mxprotocol"))
    if not mxprotocol_files:
        raise FileNotFoundError(
            f"No .mxprotocol file found in {input_path}. "
            "This does not look like an HCS.ai acquisition folder."
        )

    logger.info("Parsing files")
    files = parse_MD_plate_folder(input_path, only_2D=False)
    if files is None:
        raise FileNotFoundError(f"Could not parse HCS.ai data from {input_path}.")
    if query is not None:
        files = files.query(query).copy()

    out_experiment_dir = output_path / "experiment"

    # Index the raw CSVs to use their rows as output-metadata templates.
    stack_csv = _read_csv_metadata(input_path / "experiment_z_stack")
    proj_csv = _read_csv_metadata(input_path / "experiment")
    output_columns = (
        stack_csv.get("__columns__") or proj_csv.get("__columns__")
    )
    if output_columns is None:
        raise FileNotFoundError(f"No image_metadata_*.csv found in {input_path}.")

    stack_files = files[files["z"].notnull()]
    proj_files = files[files["z"].isnull()]

    # Decide whether the microscope's own ``experiment/`` projection can be
    # reused. Reuse requires that ``experiment/`` genuinely holds a projection
    # (detected from the ``_projection`` filename marker, not assumed) whose
    # method matches the requested type. Single-plane / center-Z acquisitions
    # keep raw ``_z{N}`` filenames and are never treated as projections.
    #
    # NOTE: the "mean" reuse path is UNVERIFIED. All available test datasets
    # only contain max projections (their ``zProjectionMechanism`` is always
    # "Max"), so we could not confirm that a mean acquisition reports "mean"
    # here. We assume it does; if the field instead reports some other value
    # the projection is simply recomputed from the z-stack, so this is safe.
    exp_is_projection = _experiment_holds_projection(proj_csv)
    microscope_types = _detect_microscope_projection_types(mxprotocol_files[0])
    can_reuse = (
        projection_type in ("max", "mean")
        and exp_is_projection
        and projection_type in microscope_types
        and not proj_files.empty
    )

    group_keys = ["well", "field", "time_point", "channel"]
    delayed_list = []
    for _, group in stack_files.groupby(group_keys, sort=False):
        row0 = group.iloc[0]
        if can_reuse:
            # Reuse the microscope's existing projection for this group.
            match = proj_files[
                (proj_files["well"] == row0["well"])
                & (proj_files["field"] == row0["field"])
                & (proj_files["time_point"] == row0["time_point"])
                & (proj_files["channel"] == row0["channel"])
            ]
            if len(match) == 1:
                proj_path = Path(match.iloc[0]["path"])
                csv_row = proj_csv.get(proj_path.name)
                if csv_row is not None:
                    proj_row = dict(csv_row)
                    proj_row["src_path"] = str(proj_path)
                    delayed_list.append(
                        dask.delayed(_copy_group)(
                            proj_row, out_experiment_dir, output_columns
                        )
                    )
                    continue
            logger.warning(
                f"No matching existing projection for {row0[group_keys].to_dict()}; "
                "recomputing from z-stack."
            )

        # Compute the projection from the z-stack planes.
        group_sorted = group.sort_values("z", key=lambda s: s.astype(int))
        plane_paths = list(group_sorted["path"])
        template_fname = Path(plane_paths[0]).name
        template_row = stack_csv.get(template_fname)
        if template_row is None:
            raise KeyError(
                f"No CSV metadata found for {template_fname} in experiment_z_stack."
            )
        # Match the microscope's convention: a projection is tagged with the
        # z-position of the middle plane (index nz // 2); a slice uses the
        # position of the extracted plane.
        n_planes = len(group_sorted)
        if projection_type == "slice":
            rep_idx = slice_index if slice_index is not None else n_planes // 2
        else:
            rep_idx = n_planes // 2
        pos_z = float(group_sorted["pos_z"].iloc[rep_idx])
        delayed_list.append(
            dask.delayed(_compute_group)(
                plane_paths,
                dict(template_row),
                pos_z,
                projection_type,
                slice_index,
                out_experiment_dir,
                output_columns,
            )
        )

    if not delayed_list:
        if exp_is_projection:
            detail = (
                f"the input only contains a "
                f"{'/'.join(sorted(microscope_types)) or 'saved'} projection "
                f"(no z-stack), so a '{projection_type}' projection cannot be "
                "computed."
            )
        else:
            detail = (
                "the input contains only single-plane / center-Z data "
                "(no z-stack), so there is nothing to project."
            )
        raise ValueError(f"No z-stack data found to project: {detail}")

    logger.info(f"Calculating and saving {len(delayed_list)} projections...")
    chunk_len = 20000
    csv_rows = []
    for i in range(0, len(delayed_list), chunk_len):
        chunk = delayed_list[i : i + chunk_len]
        logger.info(
            f"processing tiles {i}-{min(i + chunk_len, len(delayed_list))} "
            f"of {len(delayed_list)}"
        )
        csv_rows.extend(dask.compute(*chunk))

    # Write the regenerated metadata CSV, the .jdce and the .mxprotocol.
    out_experiment_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(csv_rows)[output_columns]
    out_df.to_csv(out_experiment_dir / "image_metadata_1.csv", index=False)

    _copy_jdce(input_path, out_experiment_dir)

    for mxprotocol in mxprotocol_files:
        dst = output_path / mxprotocol.name
        if mxprotocol.resolve() != dst.resolve():
            shutil.copy2(mxprotocol, dst)

    logger.info("Finished!")
