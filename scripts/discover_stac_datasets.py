#!/usr/bin/env python3
"""Search external STAC catalogs for candidate datasets to wrap as an
earth2studio data source.

Generalizes the manual per-catalog lookups (curl against Planetary Computer's
collections endpoint, dynamical.org's root catalog.json) used while building
earth2studio/data/planetary_computer.py's PlanetaryComputerLandsat source, so
future dataset discovery doesn't require repeating the same curl+jq probing
by hand.

Usage
-----
    python scripts/discover_stac_datasets.py <keyword> [--catalog NAME]

Examples
--------
    python scripts/discover_stac_datasets.py landsat
    python scripts/discover_stac_datasets.py viirs --catalog dynamical
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any

USER_AGENT = "earth2studio-dataset-discovery"

CATALOGS = {
    "planetary-computer": "https://planetarycomputer.microsoft.com/api/stac/v1/collections",
    "dynamical": "https://stac.dynamical.org/catalog.json",
}


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310  https only
        url, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read())


def search_planetary_computer(keyword: str) -> list[dict[str, Any]]:
    """Keyword-search Planetary Computer's live collection list.

    Flags whether each match exposes the STAC datacube extension
    (cube:variables), since that determines whether variable metadata can be
    read from the catalog directly or has to be hand-maintained in a Lexicon
    (see earth2studio/data/planetary_computer.py's per-source Lexicons).
    """
    data = _fetch_json(CATALOGS["planetary-computer"])
    kw = keyword.lower()
    results = []
    for collection in data.get("collections", []):
        haystack = " ".join(
            [collection.get("id", ""), collection.get("title", ""), collection.get("description", "")]
        ).lower()
        if kw not in haystack:
            continue
        extensions = collection.get("stac_extensions", [])
        results.append(
            {
                "catalog": "planetary-computer",
                "id": collection["id"],
                "title": collection.get("title", ""),
                "has_datacube_extension": any("datacube" in ext for ext in extensions),
                "extensions": extensions,
                "url": f"https://planetarycomputer.microsoft.com/dataset/{collection['id']}",
            }
        )
    return results


def search_dynamical(keyword: str) -> list[dict[str, Any]]:
    """Keyword-search dynamical.org's root STAC catalog child links."""
    catalog = _fetch_json(CATALOGS["dynamical"])
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
            {
                "catalog": "dynamical.org",
                "id": collection_id,
                "title": title,
                "url": href,
            }
        )
    return results


SEARCHERS = {
    "planetary-computer": search_planetary_computer,
    "dynamical": search_dynamical,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keyword", help="Substring to match against collection id/title/description")
    parser.add_argument(
        "--catalog",
        choices=[*SEARCHERS, "all"],
        default="all",
        help="Which catalog to search (default: all)",
    )
    args = parser.parse_args()

    catalogs = list(SEARCHERS) if args.catalog == "all" else [args.catalog]

    results = []
    for name in catalogs:
        try:
            results.extend(SEARCHERS[name](args.keyword))
        except Exception as e:  # noqa: BLE001  best-effort across catalogs
            print(f"[{name}] search failed: {e}")

    if not results:
        print(f"No matches for {args.keyword!r} in {catalogs}")
        return

    for r in results:
        print(f"[{r['catalog']}] {r['id']} — {r['title']}")
        print(f"    {r['url']}")
        if "has_datacube_extension" in r:
            print(f"    datacube extension (cube:variables): {r['has_datacube_extension']}")


if __name__ == "__main__":
    main()
