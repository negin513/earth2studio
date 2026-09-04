#!/usr/bin/env python3
"""Search and inspect external STAC catalogs for candidate datasets to wrap
as an earth2studio data source.

Generalizes the manual per-catalog probing (curl against Planetary
Computer's collections endpoint and live items, dynamical.org's root
catalog.json) used while building
earth2studio/data/planetary_computer.py's PlanetaryComputerLandsat source.

Two subcommands, matching the two separate questions that actually came up
while building that source:

    search   Does a dataset matching a keyword exist, and where?
    inspect  How hard would it be to wrap: single- or multi-asset items,
             fixed or per-acquisition-variable grid, datacube extension?

Usage
-----
    python scripts/discover_stac_datasets.py search <keyword> [--catalog NAME]
    python scripts/discover_stac_datasets.py inspect <catalog> <collection_id>

Examples
--------
    python scripts/discover_stac_datasets.py search landsat
    python scripts/discover_stac_datasets.py inspect planetary-computer landsat-c2-l2
    python scripts/discover_stac_datasets.py inspect dynamical noaa-gfs-analysis
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any

USER_AGENT = "earth2studio-dataset-discovery"

PC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DYNAMICAL_CATALOG_URL = "https://stac.dynamical.org/catalog.json"

# Number of items to sample when inspecting a Planetary Computer collection's
# grid/asset structure. Small on purpose: this is a scoping check, not a full
# survey, so it stays fast.
SAMPLE_ITEM_COUNT = 3


def _fetch_json(url: str, method: str = "GET", body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(  # noqa: S310  https only
        url,
        data=data,
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read())


@dataclass
class DatasetMatch:
    """A collection matching a search keyword."""

    catalog: str
    id: str
    title: str
    url: str

    def render(self) -> str:
        return f"[{self.catalog}] {self.id} — {self.title}\n    {self.url}"


@dataclass
class DatasetProfile:
    """Structural report on how a collection is shaped, to scope wrapper effort.

    ``fixed_grid`` and ``multi_asset`` are heuristics sampled from a handful
    of items (Planetary Computer) or read directly from declared collection
    metadata (dynamical.org). Treat them as a starting point, not a
    guarantee — always verify against the actual items you plan to fetch.
    """

    catalog: str
    id: str
    asset_keys: list[str] = field(default_factory=list)
    multi_asset: bool | None = None
    has_datacube_extension: bool = False
    grid_shapes: list[tuple[int, int]] = field(default_factory=list)
    grid_epsgs: list[int] = field(default_factory=list)
    fixed_grid: bool | None = None
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"[{self.catalog}] {self.id}"]
        lines.append(f"    asset keys: {self.asset_keys or '(none declared at collection level)'}")
        if self.multi_asset is not None:
            verdict = "multi-asset (per-band) items" if self.multi_asset else "single-asset items"
            lines.append(f"    asset layout: {verdict}")
            if self.multi_asset:
                lines.append(
                    "      -> needs a _prepare_asset_plans override, like PlanetaryComputerLandsat"
                )
        lines.append(f"    datacube extension (cube:variables): {self.has_datacube_extension}")
        if self.grid_shapes:
            lines.append(f"    sampled proj:shape: {self.grid_shapes}")
            lines.append(f"    sampled proj:epsg: {self.grid_epsgs}")
            if self.fixed_grid is False:
                lines.append(
                    "      -> grid varies per item; needs a fixed-reprojection-grid "
                    "pattern (WarpedVRT), like PlanetaryComputerLandsat"
                )
            elif self.fixed_grid:
                lines.append("      -> grid looks fixed across the sample; base class default should work")
        lines.extend(f"    note: {note}" for note in self.notes)
        return "\n".join(lines)


def search_planetary_computer(keyword: str) -> list[DatasetMatch]:
    """Keyword-search Planetary Computer's live collection list."""
    data = _fetch_json(f"{PC_API_URL}/collections")
    kw = keyword.lower()
    results = []
    for collection in data.get("collections", []):
        haystack = " ".join(
            [collection.get("id", ""), collection.get("title", ""), collection.get("description", "")]
        ).lower()
        if kw not in haystack:
            continue
        results.append(
            DatasetMatch(
                catalog="planetary-computer",
                id=collection["id"],
                title=collection.get("title", ""),
                url=f"https://planetarycomputer.microsoft.com/dataset/{collection['id']}",
            )
        )
    return results


def search_dynamical(keyword: str) -> list[DatasetMatch]:
    """Keyword-search dynamical.org's root STAC catalog child links."""
    catalog = _fetch_json(DYNAMICAL_CATALOG_URL)
    kw = keyword.lower()
    results = []
    for link in catalog.get("links", []):
        if link.get("rel") != "child":
            continue
        title = link.get("title", "")
        href = link.get("href", "")
        if kw not in title.lower() and kw not in href.lower():
            continue
        collection_id = href.rstrip("/").split("/")[-2] if "collection.json" in href else href
        results.append(
            DatasetMatch(catalog="dynamical.org", id=collection_id, title=title, url=href)
        )
    return results


def inspect_planetary_computer(collection_id: str) -> DatasetProfile:
    """Profile a Planetary Computer collection's asset and grid structure.

    Reads ``item_assets`` for the declared asset keys, then samples a few
    live items to check whether ``proj:shape``/``proj:epsg`` stay constant
    (fixed grid) or vary per acquisition — the distinction that made
    PlanetaryComputerLandsat need a custom reprojection grid instead of the
    base class's default fixed ``spatial_dims``.
    """
    collection = _fetch_json(f"{PC_API_URL}/collections/{collection_id}")
    extensions = collection.get("stac_extensions", [])
    item_assets = list(collection.get("item_assets", {}).keys())

    profile = DatasetProfile(
        catalog="planetary-computer",
        id=collection_id,
        asset_keys=item_assets,
        has_datacube_extension=any("datacube" in ext for ext in extensions),
    )

    try:
        search_result = _fetch_json(
            f"{PC_API_URL}/search",
            method="POST",
            body={"collections": [collection_id], "limit": SAMPLE_ITEM_COUNT},
        )
    except Exception as e:  # noqa: BLE001  profile is still useful without item sampling
        profile.notes.append(f"item sampling failed, collection-level metadata only: {e}")
        return profile

    items = search_result.get("features", [])
    if not items:
        profile.notes.append("no items returned for sampling; collection may be empty or query-gated")
        return profile

    sample_assets = items[0].get("assets", {})
    sample_asset_count = len(sample_assets)

    # A single asset covering multiple bands (a NetCDF/HDF container, or a
    # GeoTIFF whose own eo:bands/raster:bands lists more than one band) is
    # the simpler case and should be preferred when available — even if the
    # item *also* separately exposes per-band COGs alongside it, as GOES's
    # 'MCMIP-nc' does next to its individual 'C01_1km' etc. assets.
    combined_asset_keys = [
        key
        for key, asset in sample_assets.items()
        if len(asset.get("eo:bands", asset.get("raster:bands", []))) > 1
        or ("tiff" not in asset.get("type", "") and asset.get("roles") == ["data"])
    ]
    cog_asset_count = sum(
        1
        for asset in sample_assets.values()
        if "raster:bands" in asset and "tiff" in asset.get("type", "")
    )
    if combined_asset_keys:
        profile.multi_asset = False
        profile.notes.append(
            f"combined multi-band asset(s) available: {combined_asset_keys} — "
            "prefer these over per-band grouping if present"
        )
    else:
        profile.multi_asset = cog_asset_count > 1

    for item in items:
        shape = item.get("properties", {}).get("proj:shape")
        epsg = item.get("properties", {}).get("proj:epsg")
        if shape:
            profile.grid_shapes.append(tuple(shape))
        if epsg is not None:
            profile.grid_epsgs.append(epsg)

    if profile.grid_shapes:
        # proj:epsg is legitimately absent for non-standard projections (e.g.
        # GOES's geostationary view, declared via proj:wkt2 instead) — don't
        # let a missing epsg alone override a consistent shape.
        profile.fixed_grid = len(set(profile.grid_shapes)) == 1
        if profile.grid_epsgs and len(set(profile.grid_epsgs)) > 1:
            profile.fixed_grid = False
        if not profile.grid_epsgs:
            profile.notes.append(
                "no proj:epsg on sampled items (likely a non-standard/native projection, "
                "e.g. geostationary — check proj:wkt2 manually)"
            )
    else:
        profile.notes.append("items carry no proj:shape; grid fixedness unknown")

    if sample_asset_count == 0:
        profile.notes.append("sampled item had no assets at all — check collection manually")

    return profile


def inspect_dynamical(collection_id: str) -> DatasetProfile:
    """Profile a dynamical.org collection from its declared collection.json.

    Unlike Planetary Computer, dynamical.org always declares cube:dimensions
    and cube:variables at the collection level (see earth2studio/data/
    dynamical.py's _open), so no item sampling is needed here — the grid and
    variable set are already structured metadata, not something to infer.
    """
    catalog = _fetch_json(DYNAMICAL_CATALOG_URL)
    collection_url = None
    for link in catalog.get("links", []):
        if link.get("rel") == "child" and f"/{collection_id}/" in link.get("href", ""):
            collection_url = link["href"]
            break
    if collection_url is None:
        raise ValueError(f"Unknown dynamical.org collection {collection_id!r}")

    collection = _fetch_json(collection_url)
    dims = collection.get("cube:dimensions", {})
    variables = list(collection.get("cube:variables", {}).keys())
    assets = list(collection.get("assets", {}).keys())

    return DatasetProfile(
        catalog="dynamical.org",
        id=collection_id,
        asset_keys=variables,
        multi_asset=False,  # dynamical.org always exposes one icechunk repo per collection
        has_datacube_extension=True,  # declared unconditionally by dynamical.org collections
        fixed_grid=True,  # icechunk repos are a single fixed-shape dataset, not per-item
        notes=[f"dims: {list(dims.keys())}", f"assets: {assets}"],
    )


SEARCHERS = {
    "planetary-computer": search_planetary_computer,
    "dynamical": search_dynamical,
}
INSPECTORS = {
    "planetary-computer": inspect_planetary_computer,
    "dynamical": inspect_dynamical,
}


def _run_search(args: argparse.Namespace) -> None:
    catalogs = list(SEARCHERS) if args.catalog == "all" else [args.catalog]
    results: list[DatasetMatch] = []
    for name in catalogs:
        try:
            results.extend(SEARCHERS[name](args.keyword))
        except Exception as e:  # noqa: BLE001  best-effort across catalogs
            print(f"[{name}] search failed: {e}")

    if not results:
        print(f"No matches for {args.keyword!r} in {catalogs}")
        return
    for r in results:
        print(r.render())


def _run_inspect(args: argparse.Namespace) -> None:
    if args.catalog not in INSPECTORS:
        raise SystemExit(f"Unknown catalog {args.catalog!r}, choose from {list(INSPECTORS)}")
    profile = INSPECTORS[args.catalog](args.collection_id)
    print(profile.render())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Find candidate collections by keyword")
    search_parser.add_argument("keyword", help="Substring to match against collection id/title/description")
    search_parser.add_argument("--catalog", choices=[*SEARCHERS, "all"], default="all")
    search_parser.set_defaults(func=_run_search)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Profile a collection's asset/grid structure to scope wrapper effort"
    )
    inspect_parser.add_argument("catalog", choices=list(INSPECTORS))
    inspect_parser.add_argument("collection_id")
    inspect_parser.set_defaults(func=_run_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
