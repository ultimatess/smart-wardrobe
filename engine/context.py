"""Temporal + location context: EXIF parsing, seasons, GPS venue clusters, weather."""
import json
import math
import os
import sqlite3
import urllib.request
from datetime import date, datetime
from pathlib import Path

import exifread

from engine.taxonomy import HOT, MILD, WET

CLUSTER_RADIUS_KM = 0.5
TRAVEL_RADIUS_KM = 60.0


# ---------------------------------------------------------------- EXIF

def _ratio(v) -> float:
    return float(v.num) / float(v.den) if getattr(v, "den", 1) else 0.0


def _gps_to_deg(values, ref) -> float:
    d, m, s = (_ratio(x) for x in values)
    deg = d + m / 60 + s / 3600
    return -deg if str(ref).strip().upper() in ("S", "W") else deg


def parse_exif(path: Path) -> dict:
    """Returns {'taken_at': iso|None, 'lat': float|None, 'lon': float|None, 'time_source': str}."""
    out = {"taken_at": None, "lat": None, "lon": None, "time_source": "none"}
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
    except Exception:
        tags = {}
    for key in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
        if key in tags:
            try:
                out["taken_at"] = datetime.strptime(str(tags[key]).strip(), "%Y:%m:%d %H:%M:%S").isoformat()
                out["time_source"] = "exif"
                break
            except ValueError:
                continue
    if out["taken_at"] is None:
        out["taken_at"] = datetime.fromtimestamp(Path(path).stat().st_mtime).replace(microsecond=0).isoformat()
        out["time_source"] = "mtime"
    try:
        lat = _gps_to_deg(tags["GPS GPSLatitude"].values, tags.get("GPS GPSLatitudeRef", "N"))
        lon = _gps_to_deg(tags["GPS GPSLongitude"].values, tags.get("GPS GPSLongitudeRef", "E"))
        if not (lat == 0 and lon == 0):
            out["lat"], out["lon"] = round(lat, 6), round(lon, 6)
    except (KeyError, ZeroDivisionError, ValueError, TypeError):
        pass
    return out


# ---------------------------------------------------------------- seasons

def season_for(d: date | datetime | str, lat: float | None = None) -> str:
    """Indian-subcontinent climate calendar; southern hemisphere shifts by 6 months."""
    if isinstance(d, str):
        d = datetime.fromisoformat(d)
    m = d.month
    if lat is not None and lat < 0:
        m = (m + 5) % 12 + 1
    if m in (3, 4, 5, 6):
        return HOT
    if m in (7, 8, 9):
        return WET
    return MILD


def time_of_day(d: datetime | str) -> str:
    if isinstance(d, str):
        d = datetime.fromisoformat(d)
    h = d.hour
    return "morning" if 5 <= h < 12 else "afternoon" if h < 17 else "evening" if h < 22 else "night"


def season_from_weather(temp_c: float | None, humidity: float | None, rain: bool) -> str | None:
    if rain:
        return WET
    if temp_c is None:
        return None
    if temp_c >= 28 or (temp_c >= 25 and (humidity or 0) >= 70):
        return HOT
    return MILD


# ---------------------------------------------------------------- location clusters

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def assign_location_cluster(conn: sqlite3.Connection, lat: float | None, lon: float | None) -> str | None:
    """Greedy leader clustering against existing wear-log coordinates (500 m radius)."""
    if lat is None or lon is None:
        return None
    rows = conn.execute(
        "SELECT location_cluster, AVG(lat) la, AVG(lon) lo FROM wear_logs "
        "WHERE location_cluster IS NOT NULL AND lat IS NOT NULL GROUP BY location_cluster"
    ).fetchall()
    best = min(rows, key=lambda r: haversine_km(lat, lon, r["la"], r["lo"]), default=None)
    if best is not None and haversine_km(lat, lon, best["la"], best["lo"]) <= CLUSTER_RADIUS_KM:
        return best["location_cluster"]
    return f"loc-{len(rows) + 1}"


def venue_labels(conn: sqlite3.Connection) -> dict[str, str]:
    """Infers a human venue for each cluster: Home / Office / Travel / Venue-N.

    Home = most frequent evening/weekend cluster; Office = most frequent weekday-daytime cluster
    (other than home); anything > 60 km from home = Travel. Names the user set (venue_names) win.
    """
    rows = conn.execute(
        "SELECT location_cluster c, worn_at, lat, lon FROM wear_logs WHERE location_cluster IS NOT NULL"
    ).fetchall()
    if not rows:
        return {r["location_cluster"]: r["label"] for r in conn.execute("SELECT * FROM venue_names")}
    home_votes, office_votes, centers = {}, {}, {}
    for r in rows:
        dt = datetime.fromisoformat(r["worn_at"])
        pts = centers.setdefault(r["c"], [])  # manual wears may carry a cluster but no GPS
        if r["lat"] is not None and r["lon"] is not None:
            pts.append((r["lat"], r["lon"]))
        if dt.weekday() < 5 and 9 <= dt.hour < 18:
            office_votes[r["c"]] = office_votes.get(r["c"], 0) + 1
        else:
            home_votes[r["c"]] = home_votes.get(r["c"], 0) + 1
    home = max(home_votes, key=home_votes.get) if home_votes else None
    office_votes.pop(home, None)
    office = max(office_votes, key=office_votes.get) if office_votes and max(office_votes.values()) >= 2 else None

    labels = {}
    mean = lambda pts: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) if pts else None
    hc = mean(centers[home]) if home else None
    for c, pts in centers.items():
        cc = mean(pts)
        if c == home:
            labels[c] = "Home"
        elif c == office:
            labels[c] = "Office"
        elif hc and cc and haversine_km(*cc, *hc) > TRAVEL_RADIUS_KM:
            labels[c] = "Travel"
        else:
            labels[c] = f"Venue {c.split('-')[-1]}"
    for r in conn.execute("SELECT location_cluster, label FROM venue_names"):
        labels[r["location_cluster"]] = r["label"]
    return labels


# ---------------------------------------------------------------- weather (opt-in network)

def fetch_weather(lat: float, lon: float) -> dict | None:
    """Coarse current weather from Open-Meteo. Disabled unless WARDROBE_ALLOW_WEATHER=1.

    Only a 1-decimal (~11 km) rounded coordinate leaves the device; no photos or vectors ever do.
    """
    if os.environ.get("WARDROBE_ALLOW_WEATHER") != "1":
        return None
    url = (
        "https://api.open-meteo.com/v1/forecast?latitude={:.1f}&longitude={:.1f}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,rain"
    ).format(lat, lon)
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            cur = json.load(resp)["current"]
        return {
            "temp_c": cur["temperature_2m"],
            "humidity": cur["relative_humidity_2m"],
            "rain": (cur.get("rain") or cur.get("precipitation") or 0) > 0.1,
            "source": "open-meteo",
        }
    except Exception:
        return None
