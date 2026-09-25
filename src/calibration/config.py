"""
config.py – Lightweight helpers for the G_projection JSON schema.

All functions that were previously in trafficlab.io.trafficlab_config
are re-implemented here with zero external dependencies beyond stdlib.
"""
import json
import os


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def default_config(location_code: str = "UNKNOWN") -> dict:
    """Return a fresh G_projection config skeleton for *location_code*."""
    return {
        "meta": {
            "location_code": location_code,
            "version": "1.0",
        },
        "use_svg": False,
        "use_roi": False,
        "roi_method": "partial",
        "undistort": {
            "K": None,
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "resolution": None,
        },
        "homography": {
            "H": None,
            "anchors_list": [],
            "fov_polygon": [],
        },
        "parallax": {
            "x_cam_coords_sat": 0.0,
            "y_cam_coords_sat": 0.0,
            "z_cam_meters": 10.0,
            "px_per_meter": 1.0,
        },
        "layout_svg": {
            "A": None,
            "association_pairs": [],
        },
        "ref_method": "center_box",
        "proj_method": "down_h_2",
    }


def load_config(path: str) -> dict:
    """Load a G_projection JSON from *path* and return it as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_config(path: str, obj: dict) -> None:
    """Serialise *obj* to *path* as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def to_pretty_json(obj: dict) -> str:
    """Return *obj* as an indented JSON string (for display)."""
    return json.dumps(obj, indent=2, ensure_ascii=False)
