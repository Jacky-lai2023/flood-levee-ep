"""Fetch and cache NRFA annual-maximum (AMAX) peak-flow data for one gauging station.

Hazard data source: UK National River Flow Archive (NRFA) public API.
    https://nrfaapps.ceh.ac.uk/nrfa/ws/time-series?format=json-object&data-type=amax-flow&station=<id>

The AMAX series is the canonical input for UK flood-frequency analysis (FEH). We use
the longest available record so the GEV tail is data-constrained rather than prior-driven.

Default station 39001 = Thames at Kingston (record since 1883, ~140 annual maxima).
Run:  uv run python download.py            # default station
      uv run python download.py 54001      # Severn at Bewdley (backup)
"""

import json
import sys
from pathlib import Path

import numpy as np
import requests

NRFA_TS = "https://nrfaapps.ceh.ac.uk/nrfa/ws/time-series"
DATA_DIR = Path(__file__).parent / "data"


def fetch_amax(station: str) -> dict:
    """Return {'station_id', 'station_name', 'years', 'flow'} for one station's AMAX flow."""
    resp = requests.get(
        NRFA_TS,
        params={"format": "json-object", "data-type": "amax-flow", "station": station},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()

    # data-stream is a flat list: [date0, value0, date1, value1, ...]
    stream = payload["data-stream"]
    dates = stream[0::2]
    flows = np.asarray(stream[1::2], dtype=float)
    years = np.asarray([int(d[:4]) for d in dates], dtype=int)

    # Drop any missing/non-finite values defensively.
    ok = np.isfinite(flows) & (flows > 0)
    years, flows = years[ok], flows[ok]

    info = payload["station"]
    return {
        "station_id": info["id"],
        "station_name": info["name"],
        "latitude": info["latitude"],
        "longitude": info["longitude"],
        "years": years.tolist(),
        "flow": flows.tolist(),
    }


def main() -> None:
    station = sys.argv[1] if len(sys.argv) > 1 else "39001"
    DATA_DIR.mkdir(exist_ok=True)
    rec = fetch_amax(station)
    out = DATA_DIR / f"amax_{rec['station_id']}.json"
    out.write_text(json.dumps(rec, indent=2))
    flows = np.asarray(rec["flow"])
    print(f"Station {rec['station_id']} — {rec['station_name']}")
    print(f"  {len(flows)} annual maxima, {min(rec['years'])}-{max(rec['years'])}")
    print(f"  flow range {flows.min():.1f}-{flows.max():.1f} m3/s, mean {flows.mean():.1f}")
    print(f"  cached -> {out.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
