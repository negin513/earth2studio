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

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from earth2studio.models.da.video_healda import (
    CONV_UV_IN_SITU_TYPES,
    E2S_CHANNELS,
    N_META_FEATURES,
    PLATFORM_NAME_TO_ID,
    SENSOR_CHANNEL_OFFSET,
    TIME_LENGTH,
    TIME_STEP,
    VideoHealDA,
)

try:
    import cupy as cp
except ImportError:
    cp = None

# ---------- Constants ----------

NVAR = len(E2S_CHANNELS)  # 74
LEVEL_IN = 1
LEVEL_MODEL = 0
NPIX = 12 * 4**LEVEL_IN  # 48
NPIX_MODEL = 12 * 4**LEVEL_MODEL  # 12
IN_CHANNELS = 2
NLAT = 5
NLON = 10

REQUEST_TIME = np.array([np.datetime64("2024-01-01T12:00:00", "ns")])


# ---------- Mocks ----------


class PhooVideoHealDAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = IN_CHANNELS
        self.out_channels = NVAR
        self.level_in = LEVEL_IN
        self.level_model = LEVEL_MODEL
        self.npix = NPIX
        self.time_length = TIME_LENGTH

    def forward(self, x, t, second_of_day, day_of_year, obs_ctx, class_labels=None):
        batch = x.shape[0]
        return torch.randn(
            batch,
            self.out_channels,
            self.time_length,
            self.npix,
            device=x.device,
        )


class MockGrid:
    def ang2pix(self, lon, lat):
        return torch.zeros(lon.shape[0], dtype=torch.long, device=lon.device)


class MockRegridder:
    def __init__(self, nlat, nlon):
        self.nlat = nlat
        self.nlon = nlon

    def __call__(self, x):
        return torch.randn(
            *x.shape[:-1], self.nlat, self.nlon, dtype=x.dtype, device=x.device
        )


class MockObsContext:
    pass


def _mock_prepare_obs_context(**kwargs):
    return MockObsContext()


def _build_sensor_stats():
    return {
        "conv": {
            "means": np.zeros(8, dtype=np.float32),
            "stds": np.ones(8, dtype=np.float32),
            "raw_to_local": np.arange(9, dtype=int),
        },
        "atms": {
            "means": np.zeros(22, dtype=np.float32),
            "stds": np.ones(22, dtype=np.float32),
            "raw_to_local": np.arange(23, dtype=int),
        },
    }


def _build_model(device="cpu", lat_lon=False, **kwargs):
    with patch("earth2studio.models.da.video_healda.earth2grid") as mock_e2g:
        mock_e2g.healpix.Grid.return_value = MockGrid()
        mock_e2g.healpix.HEALPIX_PAD_XY = 0
        if lat_lon:
            mock_e2g.get_regridder.return_value = MockRegridder(NLAT, NLON)
        model = VideoHealDA(
            model=PhooVideoHealDAModel(),
            condition=torch.zeros(1, IN_CHANNELS, 1, NPIX),
            era5_mean=torch.zeros(1, NVAR, 1, 1),
            era5_std=torch.ones(1, NVAR, 1, 1),
            sensor_stats=_build_sensor_stats(),
            lat_lon=lat_lon,
            output_resolution=(NLAT, NLON),
            **kwargs,
        )
    model._grid_model = MockGrid()
    return model.to(device)


def _build_raw_conv_df(n_obs=10, request_time=None, variable="t", obs_type=120):
    if request_time is None:
        request_time = REQUEST_TIME
    df = pd.DataFrame(
        {
            "time": np.full(n_obs, request_time[0].astype("datetime64[ns]")),
            "lat": np.random.uniform(-90, 90, n_obs).astype(np.float32),
            "lon": np.random.uniform(0, 360, n_obs).astype(np.float32),
            "observation": np.random.uniform(200, 300, n_obs).astype(np.float32),
            "variable": variable,
            "type": np.full(n_obs, obs_type, dtype=np.uint16),
            "elev": np.full(n_obs, 100.0, dtype=np.float32),
            "pres": np.full(n_obs, 50000.0, dtype=np.float32),
        }
    )
    df.attrs = {"request_time": request_time}
    return df


def _build_raw_sat_df(n_obs=10, request_time=None, sensor="atms"):
    if request_time is None:
        request_time = REQUEST_TIME
    t = request_time[0].astype("datetime64[ns]")
    df = pd.DataFrame(
        {
            "time": np.full(n_obs, t),
            "lat": np.random.uniform(-90, 90, n_obs).astype(np.float32),
            "lon": np.random.uniform(0, 360, n_obs).astype(np.float32),
            "observation": np.random.uniform(200, 300, n_obs).astype(np.float32),
            "variable": sensor,
            "sensor_index": np.ones(n_obs, dtype=np.uint16),
            "satellite": "n20",
            "scan_angle": np.zeros(n_obs, dtype=np.float32),
            "satellite_za": np.full(n_obs, 30.0, dtype=np.float32),
            "solza": np.full(n_obs, 45.0, dtype=np.float32),
        }
    )
    df.attrs = {"request_time": request_time}
    return df


# ---------- Metadata tests ----------


def test_metadata_shape_and_branches():
    n = 6
    target = torch.full((n,), 1_700_000_000, dtype=torch.int64)
    time = target * 1_000_000_000
    lon = torch.linspace(0, 350, n)
    lat = torch.linspace(-80, 80, n)
    # First half conv (height valid), second half sat (height NaN)
    height = torch.tensor([100.0, 200.0, 300.0, np.nan, np.nan, np.nan])
    pressure = torch.tensor([900.0, 800.0, 700.0, np.nan, np.nan, np.nan])
    scan = torch.tensor([np.nan, np.nan, np.nan, 10.0, 20.0, 30.0])
    sat_zen = torch.tensor([np.nan, np.nan, np.nan, 30.0, 40.0, 50.0])
    sol_zen = torch.tensor([np.nan, np.nan, np.nan, 60.0, 70.0, 80.0])

    meta = VideoHealDA._compute_unified_metadata(
        target, lon, lat, time, height, pressure, scan, sat_zen, sol_zen
    )
    assert meta.shape == (n, N_META_FEATURES)
    assert not torch.isnan(meta).any()
    # Conv rows: sat-private block [10:30) must be zero
    assert torch.all(meta[:3, 10:30] == 0.0)
    # Sat rows: conv-private block [30:50) must be zero
    assert torch.all(meta[3:, 30:50] == 0.0)
    # Both private blocks carry signal for their own family
    assert torch.any(meta[:3, 30:50] != 0.0)
    assert torch.any(meta[3:, 10:30] != 0.0)
    # Shared latitude features
    lat_rad = torch.deg2rad(lat)
    assert torch.allclose(meta[:, 8], torch.sin(lat_rad), atol=1e-6)
    assert torch.allclose(meta[:, 9], torch.cos(lat_rad), atol=1e-6)
    # Zero relative time -> dt features are [0, 0, sin(0)=0, cos(0)=1]
    assert torch.allclose(meta[:, 4:7], torch.zeros(n, 3), atol=1e-6)
    assert torch.allclose(meta[:, 7], torch.ones(n), atol=1e-6)


def test_metadata_empty():
    empty = torch.empty(0)
    empty_i = torch.empty(0, dtype=torch.int64)
    meta = VideoHealDA._compute_unified_metadata(
        empty_i, empty, empty, empty_i, empty, empty, empty, empty, empty
    )
    assert meta.shape == (0, N_META_FEATURES)


# ---------- Frame assignment tests ----------


def test_assign_frames_layout():
    model = _build_model()
    request_time = REQUEST_TIME
    # One obs exactly at each frame time
    frame_times = np.array(
        [request_time[0] - (TIME_LENGTH - 1 - i) * TIME_STEP for i in range(TIME_LENGTH)]
    )
    df = _build_raw_conv_df(TIME_LENGTH, request_time)
    df["time"] = frame_times.astype("datetime64[ns]")
    obs = model.filter_and_normalize(df, None)
    obs = model.assign_frames(obs, request_time)

    assert len(obs) == TIME_LENGTH
    assert sorted(obs["frame_idx"].tolist()) == list(range(TIME_LENGTH))
    assert (obs["batch_idx"] == 0).all()
    # Each obs targets its own frame time
    expected_sec = frame_times.astype("datetime64[s]").astype(np.int64)
    assert sorted(obs["target_time_sec"].tolist()) == sorted(expected_sec.tolist())


def test_assign_frames_drops_out_of_window():
    model = _build_model()
    request_time = REQUEST_TIME
    df = _build_raw_conv_df(3, request_time)
    df["time"] = np.array(
        [
            request_time[0],  # in window (last frame)
            request_time[0] - np.timedelta64(49, "h"),  # before window
            request_time[0] + np.timedelta64(4, "h"),  # after window
        ]
    ).astype("datetime64[ns]")
    obs = model.filter_and_normalize(df, None)
    obs = model.assign_frames(obs, request_time)
    assert len(obs) == 1
    assert obs["frame_idx"].iloc[0] == TIME_LENGTH - 1


def test_assign_frames_batched():
    model = _build_model()
    request_time = np.array(
        [
            np.datetime64("2024-01-01T12:00:00", "ns"),
            np.datetime64("2024-01-01T18:00:00", "ns"),
        ]
    )
    df = _build_raw_conv_df(5, request_time)
    df["time"] = np.full(5, request_time[0])
    obs = model.filter_and_normalize(df, None)
    obs = model.assign_frames(obs, request_time)
    # Obs at batch-0's last frame are also batch-1's frame 6
    assert set(obs["batch_idx"].unique()) == {0, 1}
    b0 = obs[obs["batch_idx"] == 0]
    b1 = obs[obs["batch_idx"] == 1]
    assert (b0["frame_idx"] == TIME_LENGTH - 1).all()
    assert (b1["frame_idx"] == TIME_LENGTH - 2).all()


# ---------- Input construction tests ----------


def test_build_input_tensors():
    model = _build_model()
    request_time = REQUEST_TIME
    df = _build_raw_conv_df(10, request_time)
    obs = model.filter_and_normalize(df, None)
    obs = model.assign_frames(obs, request_time)
    inputs = model.build_input(obs, request_time)

    n_obs = len(obs)
    assert inputs["obs"].shape == (n_obs,)
    assert inputs["float_metadata"].shape == (n_obs, N_META_FEATURES)
    assert inputs["flat_idx"].dtype == torch.int32
    total_pixels = len(request_time) * TIME_LENGTH * NPIX_MODEL
    assert (inputs["flat_idx"] >= 0).all()
    assert (inputs["flat_idx"] < total_pixels).all()
    assert inputs["condition"].shape == (1, IN_CHANNELS, TIME_LENGTH, NPIX)
    assert inputs["second_of_day"].shape == (1, TIME_LENGTH)
    assert inputs["day_of_year"].shape == (1, TIME_LENGTH)
    # 12:00 analysis with 6h frames: seconds_of_day alternates 18:00/00:00/06:00/12:00
    assert inputs["second_of_day"][0, -1].item() == 12 * 3600
    assert inputs["second_of_day"][0, -2].item() == 6 * 3600


def test_global_ids():
    model = _build_model()
    request_time = REQUEST_TIME
    conv = _build_raw_conv_df(4, request_time, variable="t")
    sat = _build_raw_sat_df(4, request_time, sensor="atms")
    obs = model.filter_and_normalize(conv, sat)
    conv_rows = obs[obs["sensor"] == "conv"]
    sat_rows = obs[obs["sensor"] == "atms"]
    # Conv 't' -> local channel 5 + conv offset; platform family 't'
    assert (conv_rows["global_channel"] == SENSOR_CHANNEL_OFFSET["conv"] + 5).all()
    assert (conv_rows["global_platform"] == PLATFORM_NAME_TO_ID["t"]).all()
    # ATMS raw channel 1 -> local 0 + atms offset 0; platform n20
    assert (sat_rows["global_channel"] == 0).all()
    assert (sat_rows["global_platform"] == PLATFORM_NAME_TO_ID["n20"]).all()


# ---------- Conv filtering tests ----------


def test_conv_gps_level1_only():
    model = _build_model()
    df = _build_raw_conv_df(6, REQUEST_TIME)
    df["variable"] = ["gps", "gps_t", "gps_q", "t", "u", "v"]
    df["observation"] = [0.01, 250.0, 0.005, 250.0, 10.0, 10.0]
    df["type"] = np.array([0, 0, 0, 120, 220, 220], dtype=np.uint16)
    obs = model.filter_and_normalize(df, None)
    assert set(obs["local_channel"].unique()) == {0, 5, 6, 7}  # gps_t/gps_q dropped


def test_conv_uv_in_situ_only():
    model = _build_model()
    df = _build_raw_conv_df(4, REQUEST_TIME, variable="u")
    df["observation"] = np.full(4, 10.0)
    df["type"] = np.array(
        [CONV_UV_IN_SITU_TYPES[0], CONV_UV_IN_SITU_TYPES[1], 244, 250],
        dtype=np.uint16,
    )
    obs = model.filter_and_normalize(df, None)
    assert len(obs) == 2  # satellite-derived wind types dropped

    model_all = _build_model(uv_in_situ_only=False)
    obs_all = model_all.filter_and_normalize(df, None)
    assert len(obs_all) == 4


# ---------- End-to-end call tests ----------


@pytest.mark.parametrize("n_times", [1, 2])
def test_call_output(n_times):
    model = _build_model()
    request_time = np.array(
        [
            np.datetime64("2024-01-01T12:00:00", "ns") + i * np.timedelta64(6, "h")
            for i in range(n_times)
        ]
    )
    conv = _build_raw_conv_df(20, request_time)
    sat = _build_raw_sat_df(20, request_time)

    with patch(
        "earth2studio.models.da.video_healda.prepare_obs_context",
        _mock_prepare_obs_context,
    ):
        da = model(conv, sat)

    assert da.dims == ("time", "variable", "npix")
    assert da.shape == (n_times, NVAR, NPIX)
    assert (da["time"].values == request_time).all()
    assert list(da["variable"].values) == E2S_CHANNELS
    assert not np.isnan(np.asarray(da.data)).all()


def test_call_lat_lon_output():
    model = _build_model(lat_lon=True)
    conv = _build_raw_conv_df(20, REQUEST_TIME)

    with patch(
        "earth2studio.models.da.video_healda.prepare_obs_context",
        _mock_prepare_obs_context,
    ):
        da = model(conv, None)

    assert da.dims == ("time", "variable", "lat", "lon")
    assert da.shape == (1, NVAR, NLAT, NLON)


def test_call_no_obs_raises():
    model = _build_model()
    with pytest.raises(ValueError):
        model(None, None)


def test_call_missing_request_time_raises():
    model = _build_model()
    df = _build_raw_conv_df(5)
    df.attrs = {}
    with pytest.raises(ValueError):
        model(df, None)


def test_call_empty_after_filter_returns_nan():
    model = _build_model()
    df = _build_raw_conv_df(5, REQUEST_TIME)
    # Push all obs far outside the assimilation window
    df["time"] = np.full(
        5, REQUEST_TIME[0] - np.timedelta64(100, "h"), dtype="datetime64[ns]"
    )
    da = model(df, None)
    assert np.isnan(np.asarray(da.data)).all()


def test_generator():
    model = _build_model()
    conv = _build_raw_conv_df(10, REQUEST_TIME)
    gen = model.create_generator()
    assert next(gen) is None
    with patch(
        "earth2studio.models.da.video_healda.prepare_obs_context",
        _mock_prepare_obs_context,
    ):
        da = gen.send((conv, None))
    assert da.shape == (1, NVAR, NPIX)
    gen.close()


# ---------- Coordinate tests ----------


def test_output_coords():
    model = _build_model()
    (coords,) = model.output_coords(model.input_coords(), request_time=REQUEST_TIME)
    assert list(coords) == ["time", "variable", "npix"]
    assert len(coords["variable"]) == NVAR
    assert len(coords["npix"]) == NPIX

    (coords_nat,) = model.output_coords(model.input_coords())
    assert np.isnat(coords_nat["time"][0])


def test_input_coords_schemas():
    model = _build_model()
    conv_schema, sat_schema = model.input_coords()
    assert "time" in conv_schema
    assert "pres" in conv_schema
    assert "sensor_index" in sat_schema


def test_init_coords():
    model = _build_model()
    assert model.init_coords() is None
