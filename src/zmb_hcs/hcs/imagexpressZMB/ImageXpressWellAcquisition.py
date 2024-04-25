from pathlib import Path
from typing import Optional, Union

import numpy as np
import dask
import dask.array as da
import pandas as pd

from faim_hcs.hcs.acquisition import TileAlignmentOptions, WellAcquisition
from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff_metadata
from faim_hcs.stitching import Tile
from faim_hcs.stitching.Tile import TilePosition


class ImageXpressWellAcquisition(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        z_spacing: Optional[float],
        background_correction_matrices: dict[str, Union[Path, str]] = None,
        illumination_correction_matrices: dict[str, Union[Path, str]] = None,
    ) -> None:
        self._z_spacing = z_spacing
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _assemble_tiles(self) -> list[Tile]:
        def _load_positions(fn):
            metadata = load_metaseries_tiff_metadata(fn)
            out = np.array([
                metadata["pixel-size-y"],
                metadata["pixel-size-x"],
                int(metadata["stage-position-y"] / metadata["spatial-calibration-y"]),
                int(metadata["stage-position-x"] / metadata["spatial-calibration-x"]),
            ])
            return out
        
        fns = self._files.path.to_numpy()
        lazy_positions = [dask.delayed(_load_positions)(fn) for fn in fns]
        arrays = [
            da.from_delayed(
                lazy_position,
                dtype=int,
                shape=(4,)
            )
            for lazy_position in lazy_positions
        ]
        positions = da.stack(arrays, axis=0).compute()

        tiles = []
        for (i, row), pos in zip(self._files.iterrows(), positions):
            file = row["path"]
            time_point = 0
            channel = row["channel"]
            if self._z_spacing is None:
                z = 1
            else:
                z = row["z"] if row["z"] is not None else 1

            bgcm = None
            if self._background_correction_matrices is not None:
                bgcm = self._background_correction_matrices[channel]

            icm = None
            if self._illumination_correction_matrices is not None:
                icm = self._illumination_correction_matrices[channel]

            tiles.append(
                Tile(
                    path=file,
                    shape=(pos[0], pos[1]),
                    position=TilePosition(
                        time=time_point,
                        channel=int(channel[1:]),
                        z=z,
                        y=pos[2],
                        x=pos[3],
                    ),
                    background_correction_matrix_path=bgcm,
                    illumination_correction_matrix_path=icm,
                )
            )
        return tiles

    # def _assemble_tiles(self) -> list[Tile]:
    #     tiles = []
    #     for i, row in self._files.iterrows():
    #         file = row["path"]
    #         time_point = 0
    #         channel = row["channel"]
    #         metadata = load_metaseries_tiff_metadata(file)
    #         if self._z_spacing is None:
    #             z = 1
    #         else:
    #             z = row["z"] if row["z"] is not None else 1

    #         bgcm = None
    #         if self._background_correction_matrices is not None:
    #             bgcm = self._background_correction_matrices[channel]

    #         icm = None
    #         if self._illumination_correction_matrices is not None:
    #             icm = self._illumination_correction_matrices[channel]

    #         tiles.append(
    #             Tile(
    #                 path=file,
    #                 shape=(metadata["pixel-size-y"], metadata["pixel-size-x"]),
    #                 position=TilePosition(
    #                     time=time_point,
    #                     channel=int(channel[1:]),
    #                     z=z,
    #                     y=int(
    #                         metadata["stage-position-y"]
    #                         / metadata["spatial-calibration-y"]
    #                     ),
    #                     x=int(
    #                         metadata["stage-position-x"]
    #                         / metadata["spatial-calibration-x"]
    #                     ),
    #                 ),
    #                 background_correction_matrix_path=bgcm,
    #                 illumination_correction_matrix_path=icm,
    #             )
    #         )
    #     return tiles

    def get_yx_spacing(self) -> tuple[float, float]:
        metadata = load_metaseries_tiff_metadata(self._files.iloc[0]["path"])
        return (metadata["spatial-calibration-y"], metadata["spatial-calibration-x"])

    def get_z_spacing(self) -> Optional[float]:
        return self._z_spacing

    def get_axes(self) -> list[str]:
        if "z" in self._files.columns:
            axes = ["z", "y", "x"]
        else:
            axes = ["y", "x"]

        if self._files["channel"].nunique() > 1:
            axes = ["c"] + axes

        return axes
