# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime as dt
import math
from collections import OrderedDict
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.da.base import AssimilationModel
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem, FrameSchema, TimeArray

try:
    import cupy as cp
except ImportError:
    cp = None  # type: ignore[assignment]

try:
    import cudf
except ImportError:
    cudf = None  # type: ignore[assignment, misc]

try:
    import earth2grid
    from physicsnemo.experimental.models.healda import (
        VideoHealDA as _VideoHealDAModel,
    )
    from physicsnemo.experimental.models.healda import prepare_obs_context
except ImportError:
    OptionalDependencyFailure("da-healda-v2")
    earth2grid = None
    _VideoHealDAModel = None
    prepare_obs_context = None

# Internal channel names used by the model weights / stats CSV.
ERA5_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
ERA5_VARIABLES_3D = ["U", "V", "T", "Z", "Q"]
ERA5_VARIABLES_2D = [
    "tcwv",
    "tas",
    "uas",
    "vas",
    "100u",
    "100v",
    "pres_msl",
    "sst",
    "sic",
]
ERA5_CHANNELS = [
    *[f"{var}{lev}" for var in ERA5_VARIABLES_3D for lev in ERA5_LEVELS],
    *ERA5_VARIABLES_2D,
]

# Mapping / channels based on E2S standard vocab
_CHANNEL_TO_E2S: dict[str, str] = {
    "tas": "t2m",
    "uas": "u10m",
    "vas": "v10m",
    "100u": "u100m",
    "100v": "v100m",
    "pres_msl": "msl",
}
for _var3d in ERA5_VARIABLES_3D:
    for _lev in ERA5_LEVELS:
        _CHANNEL_TO_E2S[f"{_var3d}{_lev}"] = f"{_var3d.lower()}{_lev}"
E2S_CHANNELS = [_CHANNEL_TO_E2S.get(ch, ch) for ch in ERA5_CHANNELS]

# Analysis window layout: 8 frames at 6 hour spacing, analysis = last frame.
TIME_LENGTH = 8
TIME_STEP = np.timedelta64(6, "h")
# Per-frame observation context window relative to each frame's valid time.
# Frames are 6 h apart with a +/-3 h context, so consecutive windows tile the
# timeline; each observation is assigned to its nearest frame.
FRAME_TOLERANCE = np.timedelta64(3, "h")

# Supported sensor lists / mappings. The V2 model was additionally trained with
# PCA-compressed hyperspectral sounders (iasi-pca, cris-fsr-pca, airs-pca) which
# are not yet available through earth2studio data sources; running without them
# is supported but may reduce analysis skill.
SAT_SENSORS = ("atms", "mhs", "amsua", "amsub")
ALL_SENSORS = (*SAT_SENSORS, "conv")
SENSOR_PLATFORMS: dict[str, list[str]] = {
    "atms": ["npp", "n20"],
    "mhs": ["metop-a", "metop-b", "metop-c", "n18", "n19"],
    "amsua": [
        "metop-a",
        "metop-b",
        "metop-c",
        "n15",
        "n16",
        "n17",
        "n18",
        "n19",
    ],
    "amsub": ["n15", "n16", "n17"],
}

# Unified global channel-id space: per-sensor local channel + sensor offset.
# Offsets follow the training-time sensor registry ordering and widths:
# atms(22), mhs(5), amsua(15), amsub(5), iasi(175), cris-fsr(100), conv(8),
# iasi-pca(32), cris-fsr-pca(32), airs(117), airs-pca(32), conv-plevel(92).
SENSOR_CHANNEL_OFFSET: dict[str, int] = {
    "atms": 0,
    "mhs": 22,
    "amsua": 27,
    "amsub": 42,
    "iasi": 47,
    "cris-fsr": 222,
    "conv": 322,
    "iasi-pca": 330,
    "cris-fsr-pca": 362,
    "airs": 394,
    "airs-pca": 511,
    "conv-plevel": 543,
}

# Unified global platform-id space (training-time registry).
PLATFORM_NAME_TO_ID: dict[str, int] = {
    "aqua": 0,
    "aura": 1,
    "f10": 2,
    "f11": 3,
    "f13": 4,
    "f14": 5,
    "f15": 6,
    "g08": 7,
    "g10": 8,
    "g11": 9,
    "g12": 10,
    "m08": 11,
    "m09": 12,
    "m10": 13,
    "metop-a": 14,
    "metop-b": 15,
    "metop-c": 16,
    "n11": 17,
    "n12": 18,
    "n14": 19,
    "n15": 20,
    "n16": 21,
    "n17": 22,
    "n18": 23,
    "n19": 24,
    "n20": 25,
    "npp": 26,
    "gps": 27,
    "ps": 28,
    "q": 29,
    "t": 30,
    "uv": 31,
}

# E2studio conventional variable -> (0-based local_channel_id, platform family).
# Channel ordering matches the training conv registry:
#   0: gps_angle, 1: gps_t, 2: gps_q, 3: ps, 4: q, 5: t, 6: u, 7: v
# Conventional observations carry the platform family (gps/ps/q/t/uv) as their
# global platform id; the FiLM tokenizer routes on it.
CONV_VAR_CHANNEL: dict[str, int] = {
    "gps": 0,
    "gps_t": 1,
    "gps_q": 2,
    "pres": 3,
    "q": 4,
    "t": 5,
    "u": 6,
    "v": 7,
}
CONV_VAR_PLATFORM: dict[str, str] = {
    "gps": "gps",
    "gps_t": "gps",
    "gps_q": "gps",
    "pres": "ps",
    "q": "q",
    "t": "t",
    "u": "uv",
    "v": "uv",
}
# In-situ PREPBUFR report types kept for u/v when uv_in_situ_only is enabled
# (matches the production training config).
CONV_UV_IN_SITU_TYPES = (220, 221, 229, 230, 231, 232, 233, 234, 235, 280, 282)

# Conventional QC limits
_QC_HEIGHT_MIN = 0.0
_QC_HEIGHT_MAX = 60000.0
_QC_PRESSURE_MIN_GPS = 0.5
_QC_PRESSURE_MIN_DEFAULT = 200.0
_QC_PRESSURE_MAX = 1100.0

# Per-conv-channel valid ranges (name, min_valid, max_valid)
_CONV_CHANNEL_RANGES = [
    ("gps_angle", float("-inf"), float("inf")),
    ("gps_t", 150.0, 350.0),
    ("gps_q", 0.0, 1.0),
    ("ps", float("-inf"), float("inf")),
    ("q", 0.0, 1.0),
    ("t", 150.0, 350.0),
    ("u", -100.0, 100.0),
    ("v", -100.0, 100.0),
]

# 50-dim observation metadata layout (feature v2):
#   [0:10)  shared: LST fourier(2), dt poly, dt fourier(1), lat sin/cos
#   [10:30) sat-private: scan fourier(3), sat_zen fourier(4), sol_zen fourier(3)
#   [30:50) conv-private: height fourier(5), pressure fourier(5)
# A row is conv iff height is not NaN; the off-family block stays zero.
N_META_FEATURES = 50


@check_optional_dependencies()
class VideoHealDA(torch.nn.Module, AutoModelMixin):
    """HealDA-V2 (VideoHealDA) data assimilation model for global weather analysis
    from sparse observations on a HEALPix grid.

    VideoHealDA is a stateless assimilation model that jointly processes an 8-frame,
    6-hourly (48 hour) observation window and produces a global weather analysis
    valid at the final frame. Observations are assimilated by pixel-local cross
    attention inside every transformer block of a video DiT backbone with causal
    temporal attention.

    The model accepts pre-processed observation DataFrames (from
    :py:class:`earth2studio.data.UFSObsConv` and
    :py:class:`earth2studio.data.UFSObsSat`) and produces a global analysis field.

    Note
    ----
    The V2 model was additionally trained with PCA-compressed hyperspectral
    sounder observations (IASI-PCA, CrIS-FSR-PCA, AIRS-PCA) which are not yet
    available through earth2studio data sources. Inference without them is
    supported but analysis skill may be reduced relative to the training setup.

    Parameters
    ----------
    model : torch.nn.Module
        The underlying VideoHealDA neural network
    condition : torch.Tensor
        Static conditioning fields (orography, land fraction) on the HEALPix
        level-6 padded XY grid of size [1, n_static, 1, npix]
    era5_mean : torch.Tensor
        ERA5 per-channel mean for output denormalization [1, out_variables, 1, 1]
    era5_std : torch.Tensor
        ERA5 per-channel std for output denormalization [1, out_variables, 1, 1]
    sensor_stats : dict[str, dict[str, np.ndarray]]
        Per-sensor normalization statistics loaded from the package
    lat_lon : bool, optional
        If True the model output is regridded from the native HEALPix grid to a
        regular equiangular lat-lon grid using ``earth2grid``. If False the raw
        HEALPix output is returned with an ``npix`` dimension, by default False
    output_resolution : tuple[int, int], optional
        ``(nlat, nlon)`` size of the output lat-lon grid. Only used when
        ``lat_lon=True``, by default ``(181, 360)`` (1 degree resolution)
    gps_level1_only : bool, optional
        Keep only the GPS bending-angle channel and drop the retrieved gps_t /
        gps_q channels, matching the production training configuration,
        by default True
    uv_in_situ_only : bool, optional
        Keep u/v wind observations only from in-situ PREPBUFR report types,
        matching the production training configuration, by default True

    Badges
    ------
    region:global class:da product:wind product:temp product:atmos product:sat
    product:insitu year:2026 gpu:40gb
    """

    def __init__(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        era5_mean: torch.Tensor,
        era5_std: torch.Tensor,
        sensor_stats: dict[str, dict[str, np.ndarray]],
        lat_lon: bool = False,
        output_resolution: tuple[int, int] = (181, 360),
        gps_level1_only: bool = True,
        uv_in_situ_only: bool = True,
    ) -> None:
        super().__init__()
        self._model = model
        self.register_buffer("condition", condition)
        self.register_buffer("_era5_mean", era5_mean)
        self.register_buffer("_era5_std", era5_std)
        self.register_buffer("device_buffer", torch.empty(0))
        self._sensor_stats = sensor_stats
        self._lat_lon = lat_lon
        self._gps_level1_only = gps_level1_only
        self._uv_in_situ_only = uv_in_situ_only
        # Observation packing happens on the coarse backbone grid (level_model);
        # the static condition / output state live on the fine ingest grid
        # (level_in). Both use the HEALPix padded-XY pixel ordering.
        self._grid_model = earth2grid.healpix.Grid(
            self._model.level_model, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
        )
        self._grid_in = earth2grid.healpix.Grid(
            self._model.level_in, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
        )
        self._npix_model = 12 * 4**self._model.level_model
        self._channel_stats = self._build_channel_stats()

        # Setup lat-lon regridder when requested
        if self._lat_lon:
            nlat, nlon = output_resolution
            self._output_lat = np.linspace(90, -90, nlat)
            self._output_lon = np.linspace(0, 360, nlon, endpoint=False)
            ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
            self._regridder = earth2grid.get_regridder(self._grid_in, ll_grid)
        else:
            self._output_lat = None
            self._output_lon = None
            self._regridder = None

    @property
    def device(self) -> torch.device:
        return self.device_buffer.device

    def init_coords(self) -> None:
        """Initialization coords (not required)"""
        return None

    def input_coords(self) -> tuple[FrameSchema, FrameSchema]:
        """Input coordinate system specifying required DataFrame fields.

        Returns two FrameSchemas: one for conventional observations and one for
        satellite observations. When calling the model, either may be ``None``
        but not both. Observations should cover the full 48 hour assimilation
        window ending at the analysis time (8 frames at 6 hour spacing, each
        with a +/-3 hour context).

        Returns
        -------
        tuple[FrameSchema, FrameSchema]
            (conventional_schema, satellite_schema) describing the expected
            columns for each observation DataFrame
        """
        conv_schema = FrameSchema(
            {
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lat": np.empty(0, dtype=np.float32),
                "lon": np.empty(0, dtype=np.float32),
                "observation": np.empty(0, dtype=np.float32),
                "variable": np.array(
                    ["u", "v", "q", "t", "pres", "gps", "gps_t", "gps_q"], dtype=str
                ),
                "type": np.empty(0, dtype=np.uint16),
                "elev": np.empty(0, dtype=np.float32),
                "pres": np.empty(0, dtype=np.float32),
            }
        )
        sat_schema = FrameSchema(
            {
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lat": np.empty(0, dtype=np.float32),
                "lon": np.empty(0, dtype=np.float32),
                "observation": np.empty(0, dtype=np.float32),
                "variable": np.array(list(SAT_SENSORS), dtype=str),
                "sensor_index": np.empty(0, dtype=np.uint16),
                "satellite": np.empty(0, dtype=str),
                "scan_angle": np.empty(0, dtype=np.float32),
                "satellite_za": np.empty(0, dtype=np.float32),
                "solza": np.empty(0, dtype=np.float32),
            }
        )
        return conv_schema, sat_schema

    def output_coords(
        self,
        input_coords: tuple[CoordSystem, CoordSystem],
        request_time: np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[CoordSystem]:
        """Output coordinate system for the VideoHealDA analysis.

        The model internally processes an 8-frame window but only the final
        frame — the analysis valid at ``request_time`` — is returned.

        Parameters
        ----------
        input_coords : tuple[CoordSystem]
            Input coordinate system
        request_time : np.ndarray | None, optional
            Analysis valid time(s), by default None

        Returns
        -------
        tuple[CoordSystem]
            Coordinate system with time, variable, and lat/lon or npix dimensions
        """
        if request_time is None:
            request_time = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")

        if self._lat_lon:
            return (
                CoordSystem(
                    OrderedDict(
                        {
                            "time": request_time,
                            "variable": np.array(E2S_CHANNELS, dtype=str),
                            "lat": self._output_lat,
                            "lon": self._output_lon,
                        }
                    )
                ),
            )

        return (
            CoordSystem(
                OrderedDict(
                    {
                        "time": request_time,
                        "variable": np.array(E2S_CHANNELS, dtype=str),
                        "npix": np.arange(self._model.npix),
                    }
                )
            ),
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default VideoHealDA model package from HuggingFace.

        Returns
        -------
        Package
            Model package pointing to the HuggingFace repository
        """
        # TODO: pin to the release commit SHA once the V2 checkpoint package is
        # published (packaged from the training DCP checkpoint).
        return Package(
            "hf://nvidia/healda-v2",
            cache_options={"same_names": True},
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        lat_lon: bool = False,
        output_resolution: tuple[int, int] = (181, 360),
    ) -> AssimilationModel:
        """Load VideoHealDA model from package.

        Parameters
        ----------
        package : Package
            Package containing model checkpoint and statistics
        lat_lon : bool, optional
            If True the output is regridded to a regular lat-lon grid,
            by default False
        output_resolution : tuple[int, int], optional
            ``(nlat, nlon)`` size of the output lat-lon grid. Only used when
            ``lat_lon=True``, by default ``(181, 360)``

        Returns
        -------
        AssimilationModel
            Loaded VideoHealDA assimilation model
        """
        model = _VideoHealDAModel.from_checkpoint(
            package.resolve("video_healda_ufs_era5.mdlus")
        )
        model.eval()

        # Static conditioning (orography, land fraction) on the level-6 HEALPix
        # padded XY grid, pre-normalized at packaging time
        static = np.load(package.resolve("static/static_condition_hpx6_padxy.npz"))
        condition = torch.from_numpy(static["condition"])  # [1, 2, 1, npix]

        # Load sensor normalization statistics
        sensor_stats: dict[str, dict[str, np.ndarray]] = {}
        for sensor in ALL_SENSORS:
            df = pd.read_csv(package.resolve(f"stats/{sensor}_normalizations.csv"))
            df = df[df["Platform_ID"] == -1].sort_values("Raw_Channel_ID")
            means = df["obs_mean"].to_numpy(dtype=np.float32)
            stds = df["obs_std"].to_numpy(dtype=np.float32)
            raw_ids = df["Raw_Channel_ID"].to_numpy()
            max_raw = int(raw_ids.max())
            lut = np.full(max_raw + 1, 0, dtype=int)
            for local_idx, raw in enumerate(raw_ids, start=1):
                lut[int(raw)] = local_idx
            sensor_stats[sensor] = {"means": means, "stds": stds, "raw_to_local": lut}

        # Load ERA normalization stats
        stats = pd.read_csv(package.resolve("stats/era5_13_levels_stats.csv"))
        level = stats["level"].astype(int)
        channel = np.where(
            level.eq(-1), stats["variable"], stats["variable"] + level.astype(str)
        )
        ordered = stats.assign(channel=channel).set_index("channel").loc[ERA5_CHANNELS]
        era5_mean = torch.from_numpy(ordered["mean"].to_numpy(dtype=np.float32))
        era5_std = torch.from_numpy(ordered["std"].to_numpy(dtype=np.float32))

        return cls(
            model=model,
            condition=condition,
            era5_mean=era5_mean.view(1, -1, 1, 1),
            era5_std=era5_std.view(1, -1, 1, 1),
            sensor_stats=sensor_stats,
            lat_lon=lat_lon,
            output_resolution=output_resolution,
        )

    @torch.inference_mode()
    def _forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = inputs["second_of_day"].shape[0]
        noise_labels = torch.zeros([batch_size], device=self.device)

        obs_ctx = prepare_obs_context(
            obs=inputs["obs"],
            float_metadata=inputs["float_metadata"],
            obs_type=inputs["obs_type"],
            channel=inputs["global_channel"],
            platform=inputs["global_platform"],
            flat_idx=inputs["flat_idx"],
            total_pixels=batch_size * TIME_LENGTH * self._npix_model,
        )

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            prediction = self._model(
                inputs["condition"],
                noise_labels,
                inputs["second_of_day"],
                inputs["day_of_year"],
                obs_ctx,
            )

        # Denormalize: prediction is [batch, channels, time, npix]
        return self._era5_std * prediction + self._era5_mean

    def __call__(
        self,
        conv_obs: pd.DataFrame | None = None,
        sat_obs: pd.DataFrame | None = None,
    ) -> xr.DataArray:
        """Run VideoHealDA inference from conventional and/or satellite observations.

        At least one of the two observation DataFrames must be provided. Each
        DataFrame must carry a ``request_time`` entry in its ``.attrs``.
        Observations should cover the 48 hour window ending at ``request_time``;
        observations outside the window are dropped.

        Parameters
        ----------
        conv_obs : pd.DataFrame | None, optional
            Conventional observation DataFrame from
            :py:class:`earth2studio.data.UFSObsConv`, by default None
        sat_obs : pd.DataFrame | None, optional
            Satellite observation DataFrame from
            :py:class:`earth2studio.data.UFSObsSat`, by default None

        Returns
        -------
        xr.DataArray
            Global analysis valid at ``request_time`` on the HEALPix grid with
            dimensions [time, variable, npix] (or [time, variable, lat, lon]
            when ``lat_lon=True``). Data is on the same device as the model
            (cupy array for GPU, numpy for CPU).

        Raises
        ------
        ValueError
            If both *conv_obs* and *sat_obs* are ``None``
        """
        if conv_obs is None and sat_obs is None:
            raise ValueError("At least one of conv_obs or sat_obs must be provided.")

        # Determine request_time from whichever DataFrame is present
        request_time = None
        for df in (conv_obs, sat_obs):
            if df is not None:
                request_time = df.attrs.get("request_time", None)
                if request_time is not None:
                    break
        if request_time is None:
            raise ValueError(
                "Observation DataFrame must have 'request_time' in attrs. "
                "This is typically set by earth2studio data sources."
            )

        # Normalize request_time to a numpy datetime64 array
        if isinstance(request_time, np.ndarray):
            request_time = request_time.astype("datetime64[ns]")
        else:
            request_time = np.array(
                [np.datetime64(request_time, "ns")], dtype="datetime64[ns]"
            )

        # Pre-process, QC, normalize
        obs = self.filter_and_normalize(conv_obs, sat_obs)

        # Assign observations to (batch, frame) slots of each analysis window
        obs = self.assign_frames(obs, request_time)

        if len(obs) == 0:
            logger.warning("No observations after filtering, returning empty analysis")
            (output_coords,) = self.output_coords(
                self.input_coords(), request_time=request_time
            )
            return self._empty_output(output_coords)

        inputs = self.build_input(obs, request_time)
        prediction = self._forward(inputs)

        (output_coords,) = self.output_coords(
            self.input_coords(), request_time=request_time
        )
        return self.build_output(prediction, output_coords)

    def create_generator(self) -> Generator[
        xr.DataArray,
        tuple[pd.DataFrame | None, pd.DataFrame | None],
        None,
    ]:
        """Creates a generator which accepts collection of input observations and
        yields the output global assimilated data.

        Yields
        ------
        xr.DataArray
            Global analysis on the HEALPix grid

        Receives
        --------
        tuple[pd.DataFrame | None, pd.DataFrame | None]
            A ``(conv_obs, sat_obs)`` tuple sent via ``generator.send()``.
            Either element may be ``None`` but not both.
        """
        inputs = yield None  # type: ignore[misc]
        try:
            while True:
                conv_obs, sat_obs = inputs if inputs is not None else (None, None)
                da = self.__call__(conv_obs, sat_obs)
                inputs = yield da
        except GeneratorExit:
            logger.debug("VideoHealDA generator clean up complete.")

    def _build_channel_stats(self) -> pd.DataFrame:
        """Build per-(sensor, local_channel) normalization stats table."""
        rows: list[dict] = []
        for sensor_name in ALL_SENSORS:
            if sensor_name not in self._sensor_stats:
                continue
            stats = self._sensor_stats[sensor_name]
            if sensor_name == "conv":
                rows.extend(
                    {
                        "sensor": sensor_name,
                        "local_channel": ch_id,
                        "mean": float(stats["means"][ch_id]),
                        "std": float(stats["stds"][ch_id]),
                        "min_valid": min_v,
                        "max_valid": max_v,
                    }
                    for ch_id, (_, min_v, max_v) in enumerate(_CONV_CHANNEL_RANGES)
                )
            else:
                rows.extend(
                    {
                        "sensor": sensor_name,
                        "local_channel": ch_id,
                        "mean": float(stats["means"][ch_id]),
                        "std": float(stats["stds"][ch_id]),
                        "min_valid": 0.0,
                        "max_valid": 400.0,
                    }
                    for ch_id in range(len(stats["means"]))
                )
        return pd.DataFrame(rows).astype({"local_channel": np.int32})

    def prep_sat_sensor(self, df: pd.DataFrame, sensor: str) -> pd.DataFrame:
        """Standardize a satellite DataFrame into the unified obs schema.

        Parameters
        ----------
        df : pd.DataFrame
            Raw satellite observation DataFrame from UFSObsSat
        sensor : str
            Sensor name (atms, mhs, amsua, amsub)

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with unified column schema
        """
        stats = self._sensor_stats[sensor]
        if "sensor_index" not in df.columns:
            raise ValueError("Satellite observations must include 'sensor_index'.")
        raw_ch = df["sensor_index"].values.astype(int)
        platforms = SENSOR_PLATFORMS.get(sensor, [])

        unknown_vars = set(df["satellite"].unique()) - set(platforms)
        if unknown_vars:
            raise ValueError(f"Unknown satellite platform(s) present: {unknown_vars}")

        max_raw = len(stats["raw_to_local"]) - 1
        out_of_bounds = raw_ch[raw_ch > max_raw]
        if len(out_of_bounds) > 0:
            raise ValueError(
                f"Sensor {sensor!r}: sensor_index values {set(out_of_bounds.tolist())} "
                f"exceed max channel {max_raw} in stats table"
            )

        local_channel = (stats["raw_to_local"][raw_ch] - 1).astype(np.int32)
        return pd.DataFrame(
            {
                "lat": df["lat"].values.astype(np.float32),
                "lon": df["lon"].values.astype(np.float32),
                "obs_time_ns": df["time"].values.astype("datetime64[ns]"),
                "observation": df["observation"].values.astype(np.float64),
                "local_channel": local_channel,
                "global_channel": (
                    local_channel + SENSOR_CHANNEL_OFFSET[sensor]
                ).astype(np.int64),
                "global_platform": df["satellite"]
                .map(PLATFORM_NAME_TO_ID)
                .values.astype(np.int64),
                "sensor": sensor,
                "obs_type": np.int64(0),
                "height": np.float32(np.nan),
                "pressure": np.float32(np.nan),
                "scan_angle": df["scan_angle"].values.astype(np.float32),
                "sat_zenith_angle": df["satellite_za"].values.astype(np.float32),
                "sol_zenith_angle": df["solza"].values.astype(np.float32),
            }
        )

    def prep_conv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize a conventional observation DataFrame into the unified obs schema.

        Applies the production training-config filters: GPS level-1 only
        (bending angle; retrieved gps_t / gps_q dropped) and in-situ-only u/v
        report types, when enabled.

        Parameters
        ----------
        df : pd.DataFrame
            Raw conventional observation DataFrame from UFSObsConv

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with unified column schema
        """
        unknown_vars = set(df["variable"].unique()) - set(CONV_VAR_CHANNEL.keys())
        if unknown_vars:
            raise ValueError(f"Unknown conventional variable(s): {unknown_vars}")

        variable = df["variable"].values
        keep = np.ones(len(df), dtype=bool)
        if self._gps_level1_only:
            keep &= ~np.isin(variable, ["gps_t", "gps_q"])
        if self._uv_in_situ_only:
            is_uv = np.isin(variable, ["u", "v"])
            in_situ = np.isin(
                df["type"].fillna(0).values.astype(np.int64),
                CONV_UV_IN_SITU_TYPES,
            )
            keep &= ~is_uv | in_situ
        df = df.iloc[keep]

        # The model was trained with pressure values in hPa, while Earth2Studio
        # conventional data sources provide pressure-like fields in Pa.
        obs = df["observation"].values.astype(np.float64)
        is_pres_obs = df["variable"].values == "pres"
        obs[is_pres_obs] /= 100.0  # Pa -> hPa

        pressure = df["pres"].values.astype(np.float32) / np.float32(100.0)  # Pa -> hPa

        local_channel = (
            df["variable"].map(CONV_VAR_CHANNEL).values.astype(np.int32)
        )
        return pd.DataFrame(
            {
                "lat": df["lat"].values.astype(np.float32),
                "lon": df["lon"].values.astype(np.float32),
                "obs_time_ns": df["time"].values.astype("datetime64[ns]"),
                "observation": obs,
                "local_channel": local_channel,
                "global_channel": (
                    local_channel + SENSOR_CHANNEL_OFFSET["conv"]
                ).astype(np.int64),
                "global_platform": df["variable"]
                .map(CONV_VAR_PLATFORM)
                .map(PLATFORM_NAME_TO_ID)
                .values.astype(np.int64),
                "sensor": "conv",
                "obs_type": df["type"].fillna(0).values.astype(np.int64),
                "height": df["elev"].values.astype(np.float32),
                "pressure": pressure,
                "scan_angle": np.float32(np.nan),
                "sat_zenith_angle": np.float32(np.nan),
                "sol_zenith_angle": np.float32(np.nan),
            }
        )

    def filter_and_normalize(
        self,
        conv_obs: pd.DataFrame | None,
        sat_obs: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Pre-process, QC-filter, and z-score normalize observations.

        This method handles the time-independent part of the observation
        pipeline:

        1. Convert raw inputs via :meth:`prep_conv` / :meth:`prep_sat_sensor`.
        2. Merge per-channel normalization statistics.
        3. Apply QC filters (range checks, height/pressure bounds for conv).
        4. Z-score normalize observations.

        Parameters
        ----------
        conv_obs : pd.DataFrame | None
            Raw conventional observation DataFrame (or ``None``)
        sat_obs : pd.DataFrame | None
            Raw satellite observation DataFrame (or ``None``)

        Returns
        -------
        pd.DataFrame
            Unified, QC-filtered, normalized observation DataFrame
        """
        # 1. Standardize raw inputs into a unified schema.
        parts: list[pd.DataFrame] = []
        if conv_obs is not None and len(conv_obs) > 0:
            parts.append(self.prep_conv(conv_obs))
        if sat_obs is not None and len(sat_obs) > 0:
            for sensor in SAT_SENSORS:
                sensor_df = sat_obs[sat_obs["variable"] == sensor]
                if len(sensor_df) > 0:
                    parts.append(self.prep_sat_sensor(sensor_df, sensor))

        if not parts:
            return self._empty_obs_frame()

        obs = pd.concat(parts, ignore_index=True)

        # Convert cudf to pandas if needed
        if cudf is not None and isinstance(obs, cudf.DataFrame):
            obs = obs.to_pandas()

        # 2. QC and normalize once on the full set (time-independent).
        obs = obs.merge(self._channel_stats, on=["sensor", "local_channel"], how="left")
        valid = obs["observation"].notna()
        valid &= (obs["observation"] >= obs["min_valid"]) & (
            obs["observation"] <= obs["max_valid"]
        )

        # Conv-specific: height and pressure physical bounds
        is_conv = obs["sensor"] == "conv"
        if is_conv.any():
            is_gps = obs["local_channel"] <= 2
            pres_min = np.where(
                is_conv & is_gps,
                _QC_PRESSURE_MIN_GPS,
                _QC_PRESSURE_MIN_DEFAULT,
            )
            height_ok = (
                obs["height"].notna()
                & (obs["height"] >= _QC_HEIGHT_MIN)
                & (obs["height"] <= _QC_HEIGHT_MAX)
            )
            pressure_ok = (
                obs["pressure"].notna()
                & (obs["pressure"] >= pres_min)
                & (obs["pressure"] <= _QC_PRESSURE_MAX)
            )
            valid &= ~is_conv | (height_ok & pressure_ok)

        obs = obs[valid].copy()
        obs["observation"] = ((obs["observation"] - obs["mean"]) / obs["std"]).astype(
            np.float32
        )
        return obs.drop(columns=["mean", "std", "min_valid", "max_valid"])

    @staticmethod
    def _empty_obs_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lat": np.empty(0, dtype=np.float32),
                "lon": np.empty(0, dtype=np.float32),
                "obs_time_ns": np.empty(0, dtype="datetime64[ns]"),
                "observation": np.empty(0, dtype=np.float32),
                "local_channel": np.empty(0, dtype=np.int32),
                "global_channel": np.empty(0, dtype=np.int64),
                "global_platform": np.empty(0, dtype=np.int64),
                "sensor": np.empty(0, dtype=str),
                "obs_type": np.empty(0, dtype=np.int64),
                "height": np.empty(0, dtype=np.float32),
                "pressure": np.empty(0, dtype=np.float32),
                "scan_angle": np.empty(0, dtype=np.float32),
                "sat_zenith_angle": np.empty(0, dtype=np.float32),
                "sol_zenith_angle": np.empty(0, dtype=np.float32),
            }
        )

    def assign_frames(
        self,
        obs: pd.DataFrame,
        request_time: TimeArray,
    ) -> pd.DataFrame:
        """Assign each observation to a (batch, frame) slot of the analysis windows.

        Each analysis window consists of ``TIME_LENGTH`` frames at ``TIME_STEP``
        spacing ending at the request time. An observation belongs to its
        nearest frame when within ``FRAME_TOLERANCE`` of it; observations
        outside every frame window are dropped. Observations are duplicated
        across batch elements (each request time forms an independent window).

        Parameters
        ----------
        obs : pd.DataFrame
            Unified, normalized observation DataFrame
        request_time : TimeArray
            Analysis valid times (one batch element each)

        Returns
        -------
        pd.DataFrame
            Observations with ``batch_idx``, ``frame_idx`` and per-observation
            ``target_time_sec`` (frame valid time, epoch seconds) columns
        """
        if len(obs) == 0:
            out = obs.copy()
            out["batch_idx"] = np.empty(0, dtype=np.int64)
            out["frame_idx"] = np.empty(0, dtype=np.int64)
            out["target_time_sec"] = np.empty(0, dtype=np.int64)
            return out

        obs_ns = obs["obs_time_ns"].values.astype("datetime64[ns]").astype(np.int64)
        step_ns = TIME_STEP.astype("timedelta64[ns]").astype(np.int64)
        tol_ns = FRAME_TOLERANCE.astype("timedelta64[ns]").astype(np.int64)

        parts: list[pd.DataFrame] = []
        for b_idx, t in enumerate(request_time):
            t_ns = np.datetime64(t, "ns").astype(np.int64)
            first_frame_ns = t_ns - (TIME_LENGTH - 1) * step_ns
            # Nearest frame index, then keep only obs within tolerance
            frame_idx = np.round((obs_ns - first_frame_ns) / step_ns).astype(np.int64)
            frame_idx = np.clip(frame_idx, 0, TIME_LENGTH - 1)
            frame_ns = first_frame_ns + frame_idx * step_ns
            within = np.abs(obs_ns - frame_ns) <= tol_ns
            if not within.any():
                continue
            part = obs.iloc[within].copy()
            part["batch_idx"] = np.int64(b_idx)
            part["frame_idx"] = frame_idx[within]
            part["target_time_sec"] = frame_ns[within] // 1_000_000_000
            parts.append(part)

        if not parts:
            out = obs.iloc[:0].copy()
            out["batch_idx"] = np.empty(0, dtype=np.int64)
            out["frame_idx"] = np.empty(0, dtype=np.int64)
            out["target_time_sec"] = np.empty(0, dtype=np.int64)
            return out
        return pd.concat(parts, ignore_index=True)

    def build_input(
        self,
        obs: pd.DataFrame,
        request_time: TimeArray,
    ) -> dict[str, torch.Tensor]:
        """Convert the frame-assigned observation DataFrame into model-ready tensors.

        Observation order is irrelevant: ``prepare_obs_context`` sorts by the
        flat pixel index, which is computed on the coarse backbone grid as
        ``(batch_idx * TIME_LENGTH + frame_idx) * npix_model + pix``.

        Parameters
        ----------
        obs : pd.DataFrame
            Observations with ``batch_idx`` / ``frame_idx`` / ``target_time_sec``
        request_time : TimeArray
            Analysis valid times

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of tensors ready for the internal model forward call.
        """
        n_batch = len(request_time)

        def to_dev(col: str, dtype: torch.dtype | None = None) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(obs[col].values))
            if dtype is not None:
                t = t.to(dtype)
            return t.to(self.device, non_blocking=True)

        lat = to_dev("lat")
        lon = to_dev("lon")
        obs_time_int = (
            obs["obs_time_ns"].values.astype("datetime64[ns]").astype(np.int64)
        )
        obs_time = torch.from_numpy(obs_time_int).to(self.device, non_blocking=True)
        target_time = to_dev("target_time_sec", torch.int64)

        float_metadata = VideoHealDA._compute_unified_metadata(
            target_time,
            lon=lon,
            lat=lat,
            time=obs_time,
            height=to_dev("height"),
            pressure=to_dev("pressure"),
            scan_angle=to_dev("scan_angle"),
            sat_zenith_angle=to_dev("sat_zenith_angle"),
            sol_zenith_angle=to_dev("sol_zenith_angle"),
        )

        # Flat pixel index on the coarse backbone grid
        pix = self._grid_model.ang2pix(lon, lat).long()
        batch_time = (
            to_dev("batch_idx", torch.int64) * TIME_LENGTH
            + to_dev("frame_idx", torch.int64)
        )
        flat_idx = (batch_time * self._npix_model + pix).int()

        # Calendar features per frame: [batch, TIME_LENGTH]
        seconds_of_day = torch.zeros(
            n_batch, TIME_LENGTH, dtype=torch.float32, device=self.device
        )
        days_of_year = torch.zeros(
            n_batch, TIME_LENGTH, dtype=torch.float32, device=self.device
        )
        for b_idx, t in enumerate(request_time):
            for f_idx in range(TIME_LENGTH):
                t_frame = t - (TIME_LENGTH - 1 - f_idx) * TIME_STEP
                t_dt = pd.Timestamp(t_frame).to_pydatetime()
                midnight = t_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                jan1 = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0)
                seconds_of_day[b_idx, f_idx] = (t_dt - midnight).total_seconds()
                days_of_year[b_idx, f_idx] = (t_dt - jan1).total_seconds() / 86400.0

        # Static condition expanded over batch and frames:
        # [1, in_channels, 1, npix] -> [batch, in_channels, TIME_LENGTH, npix]
        condition = self.condition.expand(
            n_batch, -1, TIME_LENGTH, -1
        ).contiguous()

        return {
            "obs": to_dev("observation"),
            "float_metadata": float_metadata,
            "obs_type": to_dev("obs_type", torch.int64),
            "global_channel": to_dev("global_channel", torch.int64),
            "global_platform": to_dev("global_platform", torch.int64),
            "flat_idx": flat_idx,
            "condition": condition,
            "second_of_day": seconds_of_day,
            "day_of_year": days_of_year,
        }

    def build_output(
        self,
        prediction: torch.Tensor,
        output_coords: CoordSystem,
    ) -> xr.DataArray:
        """Convert model output tensor to xarray DataArray.

        Only the final frame of the window — the analysis valid at the request
        time — is returned.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output [batch, variable, TIME_LENGTH, npix]
        output_coords : CoordSystem
            Output coordinate system

        Returns
        -------
        xr.DataArray
            Analysis field, either on HEALPix (npix) or lat-lon grid
        """
        out = prediction[:, :, -1].contiguous()
        if self._lat_lon and self._regridder is not None:
            out = self._regridder(out.double())

        if self.device.type == "cuda" and cp is not None:
            data = cp.asarray(out)
        else:
            data = out.cpu().numpy()

        if self._lat_lon:
            return xr.DataArray(
                data=data,
                dims=["time", "variable", "lat", "lon"],
                coords=output_coords,
            )

        return xr.DataArray(
            data=data,
            dims=["time", "variable", "npix"],
            coords=output_coords,
        )

    def _empty_output(self, output_coords: CoordSystem) -> xr.DataArray:
        """Return an empty (NaN-filled) DataArray."""
        n_time = len(output_coords["time"])
        n_var = len(output_coords["variable"])
        device = self.condition.device

        if self._lat_lon:
            n_lat = len(output_coords["lat"])
            n_lon = len(output_coords["lon"])
            shape = (n_time, n_var, n_lat, n_lon)
            dims = ["time", "variable", "lat", "lon"]
        else:
            n_pix = len(output_coords["npix"])
            shape = (n_time, n_var, n_pix)  # type: ignore[assignment]
            dims = ["time", "variable", "npix"]

        data = torch.full(shape, float("nan"), dtype=torch.float32, device=device)
        if device.type == "cuda" and cp is not None:
            data_np = cp.asarray(data)
        else:
            data_np = data.cpu().numpy()

        return xr.DataArray(
            data=data_np,
            dims=dims,
            coords=output_coords,
        )

    @staticmethod
    def _fourier_features(x_norm: torch.Tensor, num_freqs: int) -> torch.Tensor:
        freqs = torch.arange(
            1, num_freqs + 1, device=x_norm.device, dtype=x_norm.dtype
        )
        x_expanded = x_norm.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)

    @staticmethod
    def _compute_unified_metadata(
        target_time_sec: torch.Tensor,
        lon: torch.Tensor,
        lat: torch.Tensor,
        time: torch.Tensor,
        height: torch.Tensor,
        pressure: torch.Tensor,
        scan_angle: torch.Tensor,
        sat_zenith_angle: torch.Tensor,
        sol_zenith_angle: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 50-dim observation metadata features (feature layout v2).

        Conv/sat specialization: height validity selects which private block
        fills slots [10:30) / [30:50); the off-family block stays zero, so every
        emitted feature carries signal (no NaN padding).

        Parameters
        ----------
        target_time_sec : torch.Tensor
            Frame valid time as epoch seconds (int64) per observation [n_obs]
        lon : torch.Tensor
            Longitude in degrees [n_obs]
        lat : torch.Tensor
            Latitude in degrees [n_obs]
        time : torch.Tensor
            Observation times as epoch nanoseconds (int64) [n_obs]
        height : torch.Tensor
            Height in meters [n_obs], NaN for satellite obs
        pressure : torch.Tensor
            Pressure in hPa [n_obs], NaN for satellite obs
        scan_angle : torch.Tensor
            Scan angle in degrees [n_obs], NaN for conventional obs
        sat_zenith_angle : torch.Tensor
            Satellite zenith angle in degrees [n_obs], NaN for conventional obs
        sol_zenith_angle : torch.Tensor
            Solar zenith angle in degrees [n_obs], NaN for conventional obs

        Returns
        -------
        torch.Tensor
            Metadata features [n_obs, 50]
        """
        device = lon.device
        n_obs = lon.shape[0]
        out = torch.zeros(n_obs, N_META_FEATURES, dtype=torch.float32, device=device)

        if n_obs == 0:
            return out

        is_conv = ~torch.isnan(height)
        two_pi = 2 * math.pi

        # Shared: local solar time Fourier(2) -> [0:4)
        sod = (time // 1_000_000_000) % 86400
        utc_hours = sod.float() / 3600.0
        lst = (utc_hours + lon / 15.0) % 24.0
        out[:, 0:4] = VideoHealDA._fourier_features(lst / 24.0 * two_pi, 2)

        # Shared: relative time polynomial -> [4:6)
        target_time_ns = target_time_sec * 1_000_000_000
        dt_days = (time - target_time_ns).float() * 1e-9 / 86400.0
        out[:, 4] = dt_days
        out[:, 5] = dt_days**2

        # Shared: relative time Fourier(1) -> [6:8)
        out[:, 6:8] = VideoHealDA._fourier_features(dt_days, 1)

        # Shared: latitude -> [8:10)
        lat_rad = torch.deg2rad(lat)
        out[:, 8] = torch.sin(lat_rad)
        out[:, 9] = torch.cos(lat_rad)

        # Sat-private [10:30): scan Fourier(3) + sat_zen Fourier(4) + sol_zen Fourier(3)
        is_sat = ~is_conv
        if is_sat.any():
            s = is_sat
            out[s, 10:16] = VideoHealDA._fourier_features(
                scan_angle[s] / 50.0 * two_pi, 3
            )
            out[s, 16:24] = VideoHealDA._fourier_features(
                sat_zenith_angle[s] / 90.0 * two_pi, 4
            )
            out[s, 24:30] = VideoHealDA._fourier_features(
                sol_zenith_angle[s] / 180.0 * two_pi, 3
            )

        # Conv-private [30:50): height Fourier(5) + pressure Fourier(5)
        if is_conv.any():
            c = is_conv
            h_norm = torch.clamp(height[c] / 60000.0, 0.0, 1.0)
            out[c, 30:40] = VideoHealDA._fourier_features(h_norm * two_pi, 5)
            p_norm = torch.clamp(pressure[c] / 1100.0, 0.0, 1.0)
            out[c, 40:50] = VideoHealDA._fourier_features(p_norm * two_pi, 5)

        return out
