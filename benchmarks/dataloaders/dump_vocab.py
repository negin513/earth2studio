#!/usr/bin/env python3
"""Dump the benchmark variable list for each data source as JSON files.

Run this with the BASELINE (0.14.0) environment: its lexicon VOCABs are a
subset of (or equal to) main's, so the baseline keys are the safe
intersection of variables supported by both versions. Sources whose lexicon
is missing from the running version are skipped with a message.

Usage: python dump_vocab.py --out-dir vars/
"""

import argparse
import json
from pathlib import Path

LEXICONS = {
    "ARCO": "ARCOLexicon",
    "WB2ERA5": "WB2Lexicon",
    "GFS": "GFSLexicon",
    "HRRR": "HRRRLexicon",
    "GEFS_FX": "GEFSLexicon",
    "NCAR_ERA5": "NCAR_ERA5Lexicon",
    "GOES": "GOESLexicon",
    "MRMS": "MRMSLexicon",
    "NClimGridDaily": "NClimGridLexicon",
    "ISD": "ISDLexicon",
    "UFSObsConv": "GSIConventionalLexicon",
    "UFSObsConv_PR914": "GSIConventionalLexicon",
    "PC_IFS": "PlanetaryComputerECMWFOpenDataIFSLexicon",
    "PC_OISST": "PlanetaryComputerOISSTLexicon",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import earth2studio.lexicon as lex

    for source, lex_name in LEXICONS.items():
        lexicon = getattr(lex, lex_name, None)
        if lexicon is None:
            print(f"{source}: SKIPPED ({lex_name} not in this earth2studio version)")
            continue
        vocab = list(lexicon.VOCAB.keys())
        (args.out_dir / f"{source}.json").write_text(json.dumps(vocab))
        print(f"{source}: {len(vocab)} variables ({lex_name})")

    # JPSS gates variables by product type (I/M/L2 can't be mixed in one call);
    # the moderate-resolution M bands are the largest coherent set (16 vars).
    jpss = getattr(lex, "JPSSLexicon", None)
    if jpss is not None:
        vocab = [k for k, v in jpss.VOCAB.items() if v[0] == "M"]
        (args.out_dir / "JPSS.json").write_text(json.dumps(vocab))
        print(f"JPSS: {len(vocab)} variables (JPSSLexicon, product_type=M)")

    # PC_IFS_5var: the 5-variable byte-range scenario from PR #1065's body
    pc_5var = ["t2m", "u10m", "z500", "q850", "stl1"]
    (args.out_dir / "PC_IFS_5var.json").write_text(json.dumps(pc_5var))
    print(f"PC_IFS_5var: {len(pc_5var)} variables (fixed subset, PR #1065 scenario)")


if __name__ == "__main__":
    main()
