"""Export the four embodiments' joint orderings as JSON for the website's joint map.

Reads the MuJoCo models the project actually trains on and writes, per robot, the
actuated joints in *model index order* together with a canonical ``side/segment/axis``
key. The website lists each robot's joints in its own order and uses the canonical key
to light up the counterpart in the other three columns -- which is what makes the
correspondence problem visible: N1 and M3 hold the same ankle joints at swapped
indices, T1 opens with a head block and has no wrists at all.

The canonical key is a *naming* correspondence, not a kinematic claim: it says two
joints carry the same name for the same segment and axis, not that they sit at the
same place on the body. Two names need a rule because the models spell them
differently, and both are recorded in ``ASSUMPTIONS`` below rather than applied
silently.

Run with the cross_bfm environment (stdlib only, but the models live there):

  python3 tools/export_joint_map.py --repo ~/Documents/Research/cross_bfm \
      --out assets/joint_map.js
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

# Model files, in the order the table lists them. These are the models whose joint
# counts match the paper's D column (29 / 27 / 23 / 23).
ROBOTS = [
    ("G1", "Unitree G1", "source", "humanoidverse/data/robots/g1/g1_29dof_mujoco.xml"),
    ("M3", "M3", "target", "humanoidverse/data/robots/m3/xmls/vr_m3.xml"),
    ("T1", "Booster T1", "target", "humanoidverse/data/robots/t1/t1_mjlab.xml"),
    ("N1", "Fourier N1", "target", "humanoidverse/data/robots/n1/n1_mjlab.xml"),
]

# Two models omit the axis on a single-axis joint. Both are flexion joints, so they
# canonicalise to pitch; without this G1's knee and elbow would read as having no
# counterpart on any other robot, which is the opposite of the truth.
ASSUMPTIONS = {
    "knee": "pitch",    # G1 'left_knee_joint'  -> left/knee/pitch
    "elbow": "pitch",   # G1 'left_elbow_joint' -> left/elbow/pitch
    "waist": "yaw",     # T1 'Waist'            -> waist/yaw
}

SEGMENTS = ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist", "head"]
GROUP_OF = {
    "hip": "leg", "knee": "leg", "ankle": "leg",
    "waist": "waist",
    "shoulder": "arm", "elbow": "arm",
    "wrist": "wrist",
    "head": "head",
}
GROUPS = ["leg", "waist", "arm", "wrist", "head"]
GROUP_LABEL = {"leg": "Leg", "waist": "Waist", "arm": "Arm", "wrist": "Wrist", "head": "Head"}
AXES = ["pitch", "roll", "yaw"]
SIDE_LABEL = {"left": "L", "right": "R", "": ""}


def parse(name: str) -> tuple[str, str, str]:
    """Canonicalise a model joint name to (side, segment, axis)."""
    n = name.lower().replace("aahead", "head")
    n = re.sub(r"_?joint$", "", n)
    n = n.replace("-", "_")

    side = ""
    if re.search(r"(^|_)left(_|$)", n) or n.startswith("left"):
        side = "left"
    elif re.search(r"(^|_)right(_|$)", n) or n.startswith("right"):
        side = "right"

    segment = next((s for s in SEGMENTS if s in n), "")
    if not segment:
        raise ValueError(f"unknown segment in joint name {name!r}")

    axis = next((a for a in AXES if a in n), "")
    if not axis:
        axis = ASSUMPTIONS.get(segment, "")
        if not axis:
            raise ValueError(f"no axis and no assumption for {name!r}")
    return side, segment, axis


def joints_of(path: str) -> list[str]:
    root = ET.parse(path).getroot()
    return [j.get("name") for j in root.iter("joint")
            if j.get("name") and j.get("type") != "free"]


def bodies_of(path: str) -> int:
    return len(list(ET.parse(path).getroot().find("worldbody").iter("body")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/Documents/Research/cross_bfm"))
    ap.add_argument("--out", default="assets/joint_map.js")
    cli = ap.parse_args()
    repo = os.path.abspath(os.path.expanduser(cli.repo))

    robots = []
    for key, label, role, rel in ROBOTS:
        path = os.path.join(repo, rel)
        names = joints_of(path)
        B = bodies_of(path)
        joints = []
        for i, name in enumerate(names):
            side, segment, axis = parse(name)
            joints.append({
                "i": i,
                "raw": name,
                "k": f"{side}/{segment}/{axis}",
                "g": GROUP_OF[segment],
                # "L hip pitch" / "waist yaw" -- short enough for four columns.
                "s": " ".join(x for x in (SIDE_LABEL[side], segment, axis) if x),
            })
        robots.append({
            "key": key, "label": label, "role": role,
            "D": len(names), "B": B, "obs": 2 * len(names) + 8 + 3 * B,
            "model": rel, "joints": joints,
        })
        print(f"[export] {key}: D={len(names)} B={B} obs={2 * len(names) + 8 + 3 * B}", flush=True)

    # A key is 'shared' when every robot has it; that is the set the encoder could
    # match on name alone. Everything else is where an embodiment differs.
    per = {r["key"]: {j["k"] for j in r["joints"]} for r in robots}
    shared = set.intersection(*per.values())
    print(f"[export] {len(shared)} of {max(len(s) for s in per.values())} keys shared by all four",
          flush=True)
    for r in robots:
        missing = sorted(k for k in set.union(*per.values()) if k not in per[r["key"]])
        print(f"[export]   {r['key']} lacks {len(missing)}: {', '.join(missing)}", flush=True)

    out = {
        "groups": GROUPS,
        "groupLabels": GROUP_LABEL,
        "assumptions": ASSUMPTIONS,
        "shared": sorted(shared),
        "robots": robots,
    }
    dest = os.path.abspath(cli.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write("window.JOINT_MAP = ")
        json.dump(out, f, separators=(",", ":"))
        f.write(";\n")
    print(f"[export] wrote {dest} ({os.path.getsize(dest) / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
