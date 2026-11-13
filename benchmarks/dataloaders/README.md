<!-- markdownlint-disable MD013 -->
# Dataloader benchmarks

Cold-cache timing of each Earth2Studio data source on the currently installed
version. Each source is asked for the largest set of variables that any model
in this repository reads from it, so the numbers reflect a realistic worst-case
model initialisation rather than the full lexicon.

## Sources, models and variable counts

A model counts as paired with a source when the repository uses them together
in `examples/`, `recipes/`, `serve/`, `docs/` or a model docstring. For each
source the model with the most variables wins; its exact variable list is in
`specs/<SOURCE>.json`, along with every other pairing found (`all_pairs`) and
the file evidence.

| Spec | Data class | Model with max variables | Variables | Other paired models (variables) | Runs by default |
|---|---|---|---|---|---|
| ARCO | `ARCO_ERA5` | UCast | 83 | GraphCastOperational 82, FuXiS2S 76, AtlasCRPS 75, SFNO 73, FCN3 72 | yes |
| NCAR_ERA5 | `NCAR_ERA5` | Aurora1p5 | 83 | SFNO 73, FCN3 72, OrbitGlobalPrecip 17 | yes |
| WB2ERA5 | `WB2ERA5` | Aurora | 69 | CBottleInfill 13, DLWP 7 | yes |
| GFS | `GFS` | SFNO | 73 | InterpCRPSDiT 73, FCN3 72, Pangu24 69, FCN 26, CorrDiffTaiwan 12, DLWP 7 | yes |
| GFS_FX | `GFS_FX` | StormCast (conditioning) | 26 | StormCastCONUS 26, StormCastSDA 26 | yes |
| HRRR | `HRRR` | StormCastCONUS | 99 | StormCast 99, StormCastSDA 99 | yes |
| IFS | `IFS` | AIFS2 | 125 | AIFS2ENS 125, AIFS 106, AIFSENS 104 | yes |
| PC_IFS | `PlanetaryComputerECMWFOpenDataIFS` | FCN3 | 72 | InterpModAFNO 72 | yes |
| GOES | `GOES` | StormScopeGOES | 8 | StormScopeMRMS 8 (conditioning) | yes |
| MRMS | `MRMS` | StormScopeMRMS | 2 | - | yes |
| ACE2 | `ACE2ERA5Data` | ACE2ERA5 | 38 | - | yes |
| CDS | `CDS_ERA5` | AIFSENS | 88 (estimated) | - | no |
| CMIP6MultiRealm | `CMIP6MultiRealm` | CorrDiffCMIP6 | 74 | - | no |
| CBottle3D | `CBottle3D` | CBottleSR | 12 | - | no |

Caveats, also recorded in each spec's `unverified` / `disabled_reason` field:

- Variable lists are the class-level defaults read from the model source. No
  model checkpoint was loaded, so lists that a checkpoint can override
  (StormCastCONUS conditioning, AIFSENS, ACE2ERA5) are not confirmed.
- AIFS-family counts on IFS exclude the inputs the model computes itself
  (cos/sin of latitude, longitude, julian day and local time, and insolation):
  AIFS2 has 134 inputs, 125 of which are read from the source.
- NCAR_ERA5: the Aurora1p5 pairing comes from a docstring only. The only
  executable pairing is FCN3 (72, `recipes/tc_tracking/cfg/helene.yaml`).
  AIFSENS is also allowed by that recipe but its input list is defined by the
  checkpoint (estimated at 88).
- CDS is disabled: it needs credentials (`~/.cdsapirc`) and its AIFSENS list is
  the same estimate.
- CMIP6MultiRealm is disabled: it needs a composite constructor (three CMIP6
  realms) that the generic harness does not build, and fetches about 60 GB.
- CBottle3D is disabled: it is a generative model that needs a GPU and a
  checkpoint, not a remote fetch.
- The table above lists the first 14 specs only. `specs/` holds 83 in total,
  covering most remote sources in `earth2studio.data`. Where no model reads a
  source, `winning_model` is null and the spec requests the source's full
  lexicon (for example GEFS_FX 214, CFS_FX 81, HRRR_FX 648).
- Specs for the ECMWF forecast variants, the Dynamical family, Meteosat/MetOp,
  CMIP6 and EarthMover were written but not all confirmed with a real fetch.
- Sample output for the default 20 sources is in `RESULTS.md`.

## Methodology

- Each trial runs `bench_source.py` in a fresh subprocess. The cache directory
  (`EARTH2STUDIO_CACHE`) is deleted first, so every trial is cold. Specs do not
  set `cache`, so the source's default caching writes into that directory.
- Timed: constructor (may do network initialisation) plus one `__call__` with
  all variables at one timestamp (and one lead time for forecast sources).
- Recorded: seconds, result shape and bytes, cache bytes, variables requested
  versus returned, and any traceback.
- Sources run sequentially so trials do not compete for bandwidth. Results
  depend on the network; use the median over trials.

## Usage

Install Earth2Studio with the data extra, then run one command. It benchmarks
the default 20 sources once each and prints the results table:

```bash
uv sync --extra data
cd benchmarks/dataloaders
uv run python driver.py
```

Other options:

```bash
python driver.py --trials 3                # median over three cold trials
python driver.py --preset all              # every enabled spec
python driver.py --sources GFS HRRR        # only the named specs
python driver.py --sources CDS             # naming a disabled spec runs it
python driver.py --dry-run                 # validate specs, no network
python make_table.py                       # re-render the table from results
```

Outputs go to `results/raw/*.json` (one per trial), `results/results.json` and
`results/table.md`. `results/` is git-ignored. Each source has a 15 minute
limit by default (`--timeout`).

The `PC_IFS` spec reads a rolling archive: update its `time` to a recent
6-hourly cycle before running.
